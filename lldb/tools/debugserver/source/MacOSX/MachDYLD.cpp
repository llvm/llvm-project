//===-- MachDYLD.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Created by Greg Clayton on 6/29/07.
//
//===----------------------------------------------------------------------===//

#include "MachDYLD.h"
#include "DNB.h"
#include "DNBDataRef.h"
#include <mach-o/loader.h>
#include "DNBLog.h"

MachDYLD::MachDYLD() :
    m_pid(INVALID_NUB_PROCESS),
    m_addr_size(4),
    m_dyld_addr(INVALID_NUB_ADDRESS),
    m_dyld_all_image_infos_addr(INVALID_NUB_ADDRESS),
    m_dylib_info_header(),
    m_current_dylibs(),
    m_changed_dylibs(),
    m_notify_break_id(INVALID_NUB_BREAK_ID),
    m_dyld_info_mutex(PTHREAD_MUTEX_RECURSIVE)
{
}

MachDYLD::~MachDYLD()
{
    Clear();
}


void
MachDYLD::Clear()
{
    PThreadMutex::Locker locker(m_dyld_info_mutex);

    nub_process_t pid = m_pid;
    if (pid != INVALID_NUB_PROCESS)
    {
        DNBProcessSetSharedLibraryInfoCallback ( pid, NULL, NULL);
        DNBBreakpointClear(pid, m_notify_break_id);
    }

    m_addr_size = 4;
    m_dyld_addr = INVALID_NUB_ADDRESS;
    m_dyld_all_image_infos_addr = INVALID_NUB_ADDRESS;
    m_dylib_info_header.Clear();
    m_current_dylibs.clear();
    m_changed_dylibs.clear();
    m_notify_break_id = INVALID_NUB_BREAK_ID;
}


void
MachDYLD::Initialize(nub_process_t pid)
{
    //printf("MachDYLD::%s(0x%4.4x)\n", __FUNCTION__, pid);
    Clear();
    m_pid = pid;
}


void
MachDYLD::ProcessStateChanged(nub_state_t state)
{
    //printf("MachDYLD::%s(%s)\n", __FUNCTION__, DNBStateAsString(state));

    switch (state)
    {
    case eStateInvalid:
    case eStateUnloaded:
    case eStateExited:
    case eStateDetached:
    case eStateAttaching:
    case eStateLaunching:
        Clear();
        break;

    case eStateStopped:
        // Keep trying find dyld each time we stop until we do
        if (!FoundDYLD())
        {
            assert(m_pid != INVALID_NUB_PROCESS);
            DNBProcessSetSharedLibraryInfoCallback ( m_pid, CopySharedInfoCallback, this);
            CheckForDYLDInMemory();
        }
        break;

    case eStateRunning:
    case eStateStepping:
    case eStateCrashed:
    case eStateSuspended:
        break;

    default:
        break;
    }
}

void
MachDYLD::SharedLibraryStateChanged(DNBExecutableImageInfo *image_infos, nub_size_t num_image_infos)
{
    //printf("MachDYLD::%s(%p, %u)\n", __FUNCTION__, image_infos, image_infos);

}

bool
MachDYLD::FoundDYLD() const
{
    return m_dyld_addr != INVALID_NUB_ADDRESS;
}

bool
MachDYLD::CheckForDYLDInMemory()
{
#if defined (__arm__)
    return CheckForDYLDInMemory(0x2fe00000);
#else
    return CheckForDYLDInMemory(0x8fe00000);
#endif
}

bool
MachDYLD::CheckForDYLDInMemory(nub_addr_t addr)
{
    std::vector<uint8_t> dyld_header;
    nub_size_t page_size = 0x1000;
    dyld_header.resize(page_size);
    nub_size_t bytes_read = DNBProcessMemoryRead(m_pid, addr, dyld_header.size(), &dyld_header[0]);
    if (bytes_read > 0)
    {
        DNBDataRef::offset_t offset = 0;
        DNBDataRef data(&dyld_header[0], bytes_read, false);
        struct mach_header *header = (struct mach_header*)data.GetData(&offset, sizeof(struct mach_header));
        if (header)
        {
            switch (header->magic)
            {
            case MH_MAGIC:
            case MH_CIGAM:
                data.SetPointerSize(4);
                m_addr_size = 4;
                break;

            case MH_MAGIC_64:
            case MH_CIGAM_64:
                data.SetPointerSize(8);
                m_addr_size = 8;
                break;

            default:
                return false;
            }

            if (header->filetype == MH_DYLINKER)
            {
            //    printf( "Found DYLD mach image at %8.8p", addr);

                m_dyld_all_image_infos_addr = DNBProcessLookupAddress(m_pid, "dyld_all_image_infos", "/usr/lib/dyld");

#if defined (__arm__)
                m_dyld_all_image_infos_addr = 0x2fe3a004;
#endif

                if (m_dyld_all_image_infos_addr != INVALID_NUB_ADDRESS)
                {
                //    printf( "Found DYLD data symbol 'dyld_all_image_infos' is %8.8p", m_dyld_all_image_infos_addr);

                    if (ReadDYLIBInfo())
                    {
                        if (m_dylib_info_header.notification != INVALID_NUB_ADDRESS)
                        {
                            m_notify_break_id = DNBBreakpointSet(m_pid, m_dylib_info_header.notification, 4, true);
                            if (NUB_BREAK_ID_IS_VALID(m_notify_break_id))
                            {
                                DNBBreakpointSetCallback(m_pid, m_notify_break_id, MachDYLD::BreakpointHit, this);
                                m_dyld_addr = addr;
                            }
                        }
                    }
                //    if (DNBLogCheckLogBit(LOG_SHLIB))
                //        Dump(DNBLogGetLogFile());
                }
                return true;
            }
        }
    }
    return false;
}

nub_bool_t
MachDYLD::BreakpointHit(nub_process_t pid, nub_thread_t tid, nub_break_t breakID, void *baton)
{
    MachDYLD *dyld = (MachDYLD*) baton;
    //printf("MachDYLD::BreakpointHit called");
    dyld->ReadDYLIBInfo();
    DNBProcessSharedLibrariesUpdated(pid);
    return false;    // Don't stop the process, let it continue
}

bool
MachDYLD::ReadDYLIBInfo()
{
    nub_addr_t addr = m_dyld_all_image_infos_addr;
    if (addr != INVALID_NUB_ADDRESS)
    {
        PThreadMutex::Locker locker(m_dyld_info_mutex);
        //printf("MachDYLD::ReadDYLIBInfo(addr =%8.8p)", addr);
        bool swap = false;
        uint32_t i = 0;
        DYLIBInfo::collection previous_dylibs;
        previous_dylibs.swap(m_current_dylibs);
        uint8_t all_dylib_info_data[32];
        nub_size_t count = 8 + m_addr_size * 2;
        nub_size_t bytes_read = DNBProcessMemoryRead(m_pid, addr, count, &all_dylib_info_data[0]);
        if (bytes_read != count)
        {
            m_dylib_info_header.Clear();
            return false;
        }

        DNBDataRef data(all_dylib_info_data, sizeof(all_dylib_info_data), swap);
        data.SetPointerSize(m_addr_size);
        DNBDataRef::offset_t offset = 0;
        m_dylib_info_header.version = data.Get32(&offset);
        m_dylib_info_header.dylib_info_count = data.Get32(&offset);
        m_dylib_info_header.dylib_info_addr = data.GetPointer(&offset);
        m_dylib_info_header.notification = data.GetPointer(&offset);
        //printf( "%s: version=%d, count=%d, addr=%8.8p, notify=%8.8p",
        //            __PRETTY_FUNCTION__,
        //            m_dylib_info_header.version,
        //            m_dylib_info_header.dylib_info_count,
        //            m_dylib_info_header.dylib_info_addr,
        //            m_dylib_info_header.notification);

        switch (m_dylib_info_header.version)
        {
        case 1:    // 10.4.x and prior
            {
            }
            break;

        case 2: // 10.5 and later
            {
            }
            break;

        default:
            //printf( "Invalid dyld all_dylib_infos version number: %d", m_dylib_info_header.version);
            return false;
            break;
        }

        // If we made it here, we are assuming that the all dylib info data should
        // be valid, lets read the info array.
        if (m_dylib_info_header.dylib_info_count > 0)
        {
            if (m_dylib_info_header.dylib_info_addr == 0)
            {
                //printf( "dyld is currently updating all_dylib_infos.");
            }
            else
            {
                m_current_dylibs.resize(m_dylib_info_header.dylib_info_count);
                count = m_current_dylibs.size() * 3 * m_addr_size;
                std::vector<uint8_t> info_data(count, 0);
                bytes_read = DNBProcessMemoryRead(m_pid, m_dylib_info_header.dylib_info_addr, count, &info_data[0]);
                if (bytes_read == count)
                {
                    DNBDataRef::offset_t info_data_offset = 0;
                    DNBDataRef info_data_ref(&info_data[0], info_data.size(), swap);
                    info_data_ref.SetPointerSize(m_addr_size);
                    for (i = 0; info_data_ref.ValidOffset(info_data_offset); i++)
                    {
                        assert (i < m_current_dylibs.size());
                        m_current_dylibs[i].address = info_data_ref.GetPointer(&info_data_offset);
                        nub_addr_t path_addr = info_data_ref.GetPointer(&info_data_offset);
                        m_current_dylibs[i].mod_date = info_data_ref.GetPointer(&info_data_offset);

                        char raw_path[PATH_MAX];
                        char resolved_path[PATH_MAX];
                        bytes_read = DNBProcessMemoryRead(m_pid, path_addr, sizeof(raw_path), (char*)&raw_path[0]);
                        if (::realpath(raw_path, resolved_path))
                            m_current_dylibs[i].path = resolved_path;
                        else
                            m_current_dylibs[i].path = raw_path;
                    }
                    assert(i == m_dylib_info_header.dylib_info_count);

                    UpdateUUIDs();
                }
                else
                {
                    //printf( "unable to read all data for all_dylib_infos.");
                    m_current_dylibs.clear();
                    return false;
                }
            }
        }
        // Read any UUID values that we can get
        if (m_current_dylibs.empty())
        {
            m_changed_dylibs = previous_dylibs;
            const size_t num_changed_dylibs = m_changed_dylibs.size();
            for (i = 0; i < num_changed_dylibs; i++)
            {
                // Indicate the shared library was unloaded by giving it an invalid
                // address...
                m_changed_dylibs[i].address = INVALID_NUB_ADDRESS;
            }
        }
        else
        {
            m_changed_dylibs.clear();

            // Most of the time when we get shared library changes, they just
            // get appended to the end of the list, so find out the min number
            // of entries in the current and previous list that match and see
            // how many are equal.
            uint32_t curr_dylib_count = m_current_dylibs.size();
            uint32_t prev_dylib_count = previous_dylibs.size();
            uint32_t common_count = std::min<uint32_t>(prev_dylib_count, curr_dylib_count);
            MachDYLD::DYLIBInfo::const_iterator curr_pos = m_current_dylibs.begin();
            MachDYLD::DYLIBInfo::const_iterator curr_end = m_current_dylibs.end();
            MachDYLD::DYLIBInfo::iterator prev_pos = previous_dylibs.begin();
            uint32_t idx;
            for (idx = 0; idx < common_count; idx++)
            {
                if (*curr_pos == *prev_pos)
                {
                    ++curr_pos;
                    ++prev_pos;
                }
                else
                    break;
            }

            // Remove all the entries that were at the exact same index and that
            // matched between the previous_dylibs and m_current_dylibs arrays. This will cover
            // most of the cases as when shared libraries get loaded they get
            // appended to the end of the list.
            if (prev_pos != previous_dylibs.begin())
            {
                previous_dylibs.erase(previous_dylibs.begin(), prev_pos);
            }

            if (previous_dylibs.empty())
            {
                // We only have new libraries to add, they are the only ones that
                // have changed.
                if (curr_pos != curr_end)
                {
                    m_changed_dylibs.assign(curr_pos, curr_end);
                }
            }
            else
            {
                // We still have items in our previous dylib list which means either
                // one or more shared libraries got unloaded somewhere in the middle
                // of the list, so we will manually search for each remaining item
                // in our current list in the previous list
                for (; curr_pos != curr_end; ++curr_pos)
                {
                    MachDYLD::DYLIBInfo::iterator pos = std::find(previous_dylibs.begin(), previous_dylibs.end(), *curr_pos);
                    if (pos == previous_dylibs.end())
                    {
                        // This dylib wasn't here before, add it to our change list
                        m_changed_dylibs.push_back(*curr_pos);
                    }
                    else
                    {
                        // This dylib was in our previous dylib list, it didn't
                        // change, so lets remove it from the previous list so we
                        // don't see it again.
                        previous_dylibs.erase(pos);
                    }
                }

                // The only items left if our previous_dylibs array will be shared
                // libraries that got unloaded (still in previous list, and not
                // mentioned in the current list).
                if (!previous_dylibs.empty())
                {
                    const size_t num_previous_dylibs = previous_dylibs.size();
                    for (i = 0; i < num_previous_dylibs; i++)
                    {
                        // Indicate the shared library was unloaded by giving it
                        // an invalid address...
                        previous_dylibs[i].address = INVALID_NUB_ADDRESS;
                    }
                    // Add all remaining previous_dylibs to the changed list with
                    // invalidated addresses so we know they got unloaded.
                    m_changed_dylibs.insert(m_changed_dylibs.end(), previous_dylibs.begin(), previous_dylibs.end());
                }
            }
        }
        return true;
    }
    return false;
}


void
MachDYLD::UpdateUUIDs()
{
    bool swap = false;
    nub_size_t page_size = 0x1000;
    uint32_t i;
    // Read any UUID values that we can get
    for (i = 0; i < m_dylib_info_header.dylib_info_count; i++)
    {
        if (!m_current_dylibs[i].UUIDValid())
        {
            std::vector<uint8_t> bytes(page_size, 0);
            nub_size_t bytes_read = DNBProcessMemoryRead(m_pid, m_current_dylibs[i].address, page_size, &bytes[0]);
            if (bytes_read > 0)
            {
                DNBDataRef::offset_t offset = 0;
                DNBDataRef data(&bytes[0], bytes_read, swap);
                struct mach_header *header = (struct mach_header*)data.GetData(&offset, sizeof(struct mach_header));
                if (header)
                {
                    switch (header->magic)
                    {
                    case MH_MAGIC:
                    case MH_CIGAM:
                        data.SetPointerSize(4);
                        m_addr_size = 4;
                        break;

                    case MH_MAGIC_64:
                    case MH_CIGAM_64:
                        data.SetPointerSize(8);
                        m_addr_size = 8;
                        offset += 4;    // Skip the extra reserved field in the 64 bit mach header
                        break;

                    default:
                        continue;
                    }

                    if (header->sizeofcmds > bytes_read)
                    {
                        bytes.resize(header->sizeofcmds);
                        nub_addr_t addr = m_current_dylibs[i].address + bytes_read;
                        bytes_read += DNBProcessMemoryRead(m_pid, addr , header->sizeofcmds - bytes_read, &bytes[bytes_read]);
                    }
                    assert(bytes_read >= header->sizeofcmds);
                    uint32_t cmd_idx;
                    DNBSegment segment;

                    for (cmd_idx = 0; cmd_idx < header->ncmds; cmd_idx++)
                    {
                        if (data.ValidOffsetForDataOfSize(offset, sizeof(struct load_command)))
                        {
                            struct load_command load_cmd;
                            DNBDataRef::offset_t load_cmd_offset = offset;
                            load_cmd.cmd = data.Get32(&offset);
                            load_cmd.cmdsize = data.Get32(&offset);
                            switch (load_cmd.cmd)
                            {
                            case LC_SEGMENT:
                                {
                                    strncpy(segment.name, data.GetCStr(&offset, 16), 16);
                                    memset(&segment.name[16], 0, DNB_MAX_SEGMENT_NAME_LENGTH - 16);
                                    segment.addr = data.Get32(&offset);
                                    segment.size = data.Get32(&offset);
                                    m_current_dylibs[i].segments.push_back(segment);
                                }
                                break;

                            case LC_SEGMENT_64:
                                {
                                    strncpy(segment.name, data.GetCStr(&offset, 16), 16);
                                    memset(&segment.name[16], 0, DNB_MAX_SEGMENT_NAME_LENGTH - 16);
                                    segment.addr = data.Get64(&offset);
                                    segment.size = data.Get64(&offset);
                                    m_current_dylibs[i].segments.push_back(segment);
                                }
                                break;

                            case LC_UUID:
                                // We found our UUID, we can stop now...
                                memcpy(m_current_dylibs[i].uuid, data.GetData(&offset, 16), 16);
                            //    if (DNBLogCheckLogBit(LOG_SHLIB))
                            //    {
                            //        DNBLogThreaded("UUID found for aii[%d]:", i);
                            //        m_current_dylibs[i].Dump(DNBLogGetLogFile());
                            //    }
                                break;

                            default:
                                break;
                            }
                            // Set offset to be the beginning of the next load command.
                            offset = load_cmd_offset + load_cmd.cmdsize;
                        }
                    }
                }
            }
        }
    }
}


nub_addr_t
MachDYLD::GetSharedLibraryHeaderAddress(const char *shlib_path) const
{
    if (!m_current_dylibs.empty() && shlib_path && shlib_path[0])
    {
        uint32_t i;
        for (i = 0; i<m_current_dylibs.size(); i++)
        {
            if (m_current_dylibs[i].path == shlib_path)
                return m_current_dylibs[i].address;
        }
    }
    return INVALID_NUB_ADDRESS;
}


nub_size_t
MachDYLD::CopySharedLibraryInfo(DYLIBInfo::collection& dylib_coll, DNBExecutableImageInfo **image_infos)
{
    if (!dylib_coll.empty())
    {
        size_t i;
        size_t total_num_segments = 0;
        size_t segment_index = 0;
        for (i = 0; i<dylib_coll.size(); i++)
        {
            total_num_segments += dylib_coll[i].segments.size();
        }
        size_t image_infos_byte_size = sizeof(DNBExecutableImageInfo) * dylib_coll.size();
        size_t all_segments_byte_size = sizeof(DNBSegment) * total_num_segments;
        size_t total_byte_size = image_infos_byte_size + all_segments_byte_size;

        // Allocate enough space to fit all of the shared library information in
        // a single buffer so consumers can free a single chunk of data when done
        uint8_t *buf = (uint8_t*)malloc (total_byte_size);

        DNBExecutableImageInfo *info = (DNBExecutableImageInfo*)buf;
        DNBSegment *all_segments = (DNBSegment*)(buf + image_infos_byte_size);
        if (info)
        {
            for (i = 0; i<dylib_coll.size(); i++)
            {
                strncpy(info[i].name, dylib_coll[i].path.c_str(), PATH_MAX);
                // NULL terminate paths that are too long (redundant for path
                // that fit, but harmless
                info[i].name[PATH_MAX-1] = '\0';
                info[i].header_addr = dylib_coll[i].address;
                info[i].state = (dylib_coll[i].address == INVALID_NUB_ADDRESS ? eShlibStateUnloaded : eShlibStateLoaded);
                memcpy(info[i].uuid, dylib_coll[i].uuid, sizeof(uuid_t));
                info[i].num_segments = dylib_coll[i].segments.size();
                if (info[i].num_segments == 0)
                {
                    info[i].segments = NULL;
                }
                else
                {
                    info[i].segments = &all_segments[segment_index];
                    memcpy(info[i].segments, &(dylib_coll[i].segments[0]), sizeof(DNBSegment) * info[i].num_segments);
                    segment_index += info[i].num_segments;
                }

            }
            // Release ownership of the shared library array to the caller
            *image_infos = info;
            return dylib_coll.size();
        }
    }
    *image_infos = NULL;
    return 0;
}



nub_size_t
MachDYLD::CopySharedInfoCallback(nub_process_t pid, struct DNBExecutableImageInfo **image_infos, nub_bool_t only_changed, void *baton)
{
    MachDYLD *dyld = (MachDYLD*) baton;

    if (only_changed)
        return dyld->CopyChangedShlibInfo(image_infos);
    else
        return dyld->CopyCurrentShlibInfo(image_infos);

    *image_infos = NULL;
    return 0;
}

nub_size_t
MachDYLD::CopyCurrentShlibInfo(DNBExecutableImageInfo **image_infos)
{
    PThreadMutex::Locker locker(m_dyld_info_mutex);
    return CopySharedLibraryInfo(m_current_dylibs, image_infos);
}


nub_size_t
MachDYLD::CopyChangedShlibInfo(DNBExecutableImageInfo **image_infos)
{
    PThreadMutex::Locker locker(m_dyld_info_mutex);
    return CopySharedLibraryInfo(m_changed_dylibs, image_infos);
}



void
MachDYLD::DYLIBInfo::Dump(FILE *f) const
{
    if (f == NULL)
        return;
    if (address == INVALID_NUB_ADDRESS)
    {
        if (UUIDValid())
        {
            fprintf(f, "UNLOADED %8.8llx %2.2X%2.2X%2.2X%2.2X-%2.2X%2.2X-%2.2X%2.2X-%2.2X%2.2X-%2.2X%2.2X%2.2X%2.2X%2.2X%2.2X %s",
                        (uint64_t)mod_date,
                        uuid[ 0], uuid[ 1], uuid[ 2], uuid[ 3],
                        uuid[ 4], uuid[ 5], uuid[ 6], uuid[ 7],
                        uuid[ 8], uuid[ 9], uuid[10], uuid[11],
                        uuid[12], uuid[13], uuid[14], uuid[15],
                        path.c_str());
        }
        else
        {
            fprintf(f, "UNLOADED %8.8llx %s", (uint64_t)mod_date, path.c_str());
        }
    }
    else
    {
        if (UUIDValid())
        {
            fprintf(f, "%8.8llx %8.8llx %2.2X%2.2X%2.2X%2.2X-%2.2X%2.2X-%2.2X%2.2X-%2.2X%2.2X-%2.2X%2.2X%2.2X%2.2X%2.2X%2.2X %s",
                        (uint64_t)address,
                        (uint64_t)mod_date,
                        uuid[ 0], uuid[ 1],    uuid[ 2], uuid[ 3],
                        uuid[ 4], uuid[ 5], uuid[ 6], uuid[ 7],
                        uuid[ 8], uuid[ 9], uuid[10], uuid[11],
                        uuid[12], uuid[13], uuid[14], uuid[15],
                        path.c_str());
        }
        else
        {
            fprintf(f, "%8.8llx %8.8llx %s", (uint64_t)address, (uint64_t)mod_date, path.c_str());
        }
    }
}

void
MachDYLD::Dump(FILE *f) const
{
    if (f == NULL)
        return;

    PThreadMutex::Locker locker(m_dyld_info_mutex);
    fprintf(f, "\n\tMachDYLD.m_dylib_info_header: version=%d, count=%d, addr=0x%llx, notify=0x%llx",
                    m_dylib_info_header.version,
                    m_dylib_info_header.dylib_info_count,
                    (uint64_t)m_dylib_info_header.dylib_info_addr,
                    (uint64_t)m_dylib_info_header.notification);
    uint32_t i;
    fprintf(f, "\n\tMachDYLD.m_current_dylibs");
    for (i = 0; i<m_current_dylibs.size(); i++)
        m_current_dylibs[i].Dump(f);
    fprintf(f, "\n\tMachDYLD.m_changed_dylibs");
    for (i = 0; i<m_changed_dylibs.size(); i++)
        m_changed_dylibs[i].Dump(f);
}

