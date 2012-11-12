//===-- ObjectContainerBSDArchive.cpp ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ObjectContainerBSDArchive.h"

#include <ar.h>

#include "lldb/Core/Stream.h"
#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Timer.h"
#include "lldb/Host/Mutex.h"
#include "lldb/Symbol/ObjectFile.h"

using namespace lldb;
using namespace lldb_private;



ObjectContainerBSDArchive::Object::Object() :
    ar_name(),
    ar_date(0),
    ar_uid(0),
    ar_gid(0),
    ar_mode(0),
    ar_size(0),
    ar_file_offset(0),
    ar_file_size(0)
{
}

void
ObjectContainerBSDArchive::Object::Clear()
{
    ar_name.Clear();
    ar_date = 0;
    ar_uid  = 0;
    ar_gid  = 0;
    ar_mode = 0;
    ar_size = 0;
    ar_file_offset = 0;
    ar_file_size = 0;
}

uint32_t
ObjectContainerBSDArchive::Object::Extract (const DataExtractor& data, uint32_t offset)
{
    size_t ar_name_len = 0;
    std::string str;
    char *err;
    str.assign ((const char *)data.GetData(&offset, 16),    16);
    if (str.find("#1/") == 0)
    {
        // If the name is longer than 16 bytes, or contains an embedded space
        // then it will use this format where the length of the name is
        // here and the name characters are after this header.
        ar_name_len = strtoul(str.c_str() + 3, &err, 10);
    }
    else
    {
        // Strip off any spaces (if the object file name contains spaces it
        // will use the extended format above).
        str.erase (str.find(' '));
        ar_name.SetCString(str.c_str());
    }

    str.assign ((const char *)data.GetData(&offset, 12),    12);
    ar_date = strtoul(str.c_str(), &err, 10);

    str.assign ((const char *)data.GetData(&offset, 6), 6);
    ar_uid  = strtoul(str.c_str(), &err, 10);

    str.assign ((const char *)data.GetData(&offset, 6), 6);
    ar_gid  = strtoul(str.c_str(), &err, 10);

    str.assign ((const char *)data.GetData(&offset, 8), 8);
    ar_mode = strtoul(str.c_str(), &err, 8);

    str.assign ((const char *)data.GetData(&offset, 10),    10);
    ar_size = strtoul(str.c_str(), &err, 10);

    str.assign ((const char *)data.GetData(&offset, 2), 2);
    if (str == ARFMAG)
    {
        if (ar_name_len > 0)
        {
            str.assign ((const char *)data.GetData(&offset, ar_name_len), ar_name_len);
            ar_name.SetCString (str.c_str());
        }
        ar_file_offset = offset;
        ar_file_size = ar_size - ar_name_len;
        return offset;
    }
    return LLDB_INVALID_INDEX32;
}

ObjectContainerBSDArchive::Archive::Archive
(
    const lldb_private::ArchSpec &arch,
    const lldb_private::TimeValue &time
) :
    m_arch (arch),
    m_time (time),
    m_objects()
{
}

ObjectContainerBSDArchive::Archive::~Archive ()
{
}

size_t
ObjectContainerBSDArchive::Archive::ParseObjects (DataExtractor &data)
{
    std::string str;
    uint32_t offset = 0;
    str.assign((const char *)data.GetData(&offset, SARMAG), SARMAG);
    if (str == ARMAG)
    {
        Object obj;
        do
        {
            offset = obj.Extract (data, offset);
            if (offset == LLDB_INVALID_INDEX32)
                break;
            uint32_t obj_idx = m_objects.size();
            m_objects.push_back(obj);
            // Insert all of the C strings out of order for now...
            m_object_name_to_index_map.Append (obj.ar_name.GetCString(), obj_idx);
            offset += obj.ar_file_size;
            obj.Clear();
        } while (data.ValidOffset(offset));

        // Now sort all of the object name pointers
        m_object_name_to_index_map.Sort ();
    }
    return m_objects.size();
}

ObjectContainerBSDArchive::Object *
ObjectContainerBSDArchive::Archive::FindObject (const ConstString &object_name)
{
    const UniqueCStringMap<uint32_t>::Entry *match = m_object_name_to_index_map.FindFirstValueForName (object_name.GetCString());
    if (match)
        return &m_objects[match->value];
    return NULL;
}


ObjectContainerBSDArchive::Archive::shared_ptr
ObjectContainerBSDArchive::Archive::FindCachedArchive (const FileSpec &file, const ArchSpec &arch, const TimeValue &time)
{
    Mutex::Locker locker(Archive::GetArchiveCacheMutex ());
    shared_ptr archive_sp;
    Archive::Map &archive_map = Archive::GetArchiveCache ();
    Archive::Map::iterator pos = archive_map.find (file);
    // Don't cache a value for "archive_map.end()" below since we might
    // delete an archive entry...
    while (pos != archive_map.end() && pos->first == file)
    {
        if (pos->second->GetArchitecture() == arch)
        {
            if (pos->second->GetModificationTime() == time)
            {
                return pos->second;
            }
            else
            {
                // We have a file at the same path with the same architecture
                // whose modification time doesn't match. It doesn't make sense
                // for us to continue to use this BSD archive since we cache only
                // the object info which consists of file time info and also the
                // file offset and file size of any contianed objects. Since
                // this information is now out of date, we won't get the correct
                // information if we go and extract the file data, so we should 
                // remove the old and outdated entry.
                archive_map.erase (pos);
                pos = archive_map.find (file);
                continue;
            }
        }
        ++pos;
    }
    return archive_sp;
}

ObjectContainerBSDArchive::Archive::shared_ptr
ObjectContainerBSDArchive::Archive::ParseAndCacheArchiveForFile
(
    const FileSpec &file,
    const ArchSpec &arch,
    const TimeValue &time,
    DataExtractor &data
)
{
    shared_ptr archive_sp(new Archive (arch, time));
    if (archive_sp)
    {
        if (archive_sp->ParseObjects (data) > 0)
        {
            Mutex::Locker locker(Archive::GetArchiveCacheMutex ());
            Archive::GetArchiveCache().insert(std::make_pair(file, archive_sp));
        }
        else
        {
            archive_sp.reset();
        }
    }
    return archive_sp;
}

ObjectContainerBSDArchive::Archive::Map &
ObjectContainerBSDArchive::Archive::GetArchiveCache ()
{
    static Archive::Map g_archive_map;
    return g_archive_map;
}

Mutex &
ObjectContainerBSDArchive::Archive::GetArchiveCacheMutex ()
{
    static Mutex g_archive_map_mutex (Mutex::eMutexTypeRecursive);
    return g_archive_map_mutex;
}


void
ObjectContainerBSDArchive::Initialize()
{
    PluginManager::RegisterPlugin (GetPluginNameStatic(),
                                   GetPluginDescriptionStatic(),
                                   CreateInstance);
}

void
ObjectContainerBSDArchive::Terminate()
{
    PluginManager::UnregisterPlugin (CreateInstance);
}


const char *
ObjectContainerBSDArchive::GetPluginNameStatic()
{
    return "object-container.bsd-archive";
}

const char *
ObjectContainerBSDArchive::GetPluginDescriptionStatic()
{
    return "BSD Archive object container reader.";
}


ObjectContainer *
ObjectContainerBSDArchive::CreateInstance
(
    const lldb::ModuleSP &module_sp,
    DataBufferSP& data_sp,
    const FileSpec *file,
    addr_t offset,
    addr_t length)
{
    DataExtractor data;
    data.SetData (data_sp, offset, length);
    if (file && data_sp && ObjectContainerBSDArchive::MagicBytesMatch(data))
    {
        Timer scoped_timer (__PRETTY_FUNCTION__,
                            "ObjectContainerBSDArchive::CreateInstance (module = %s/%s, file = %p, file_offset = 0x%8.8llx, file_size = 0x%8.8llx)",
                            module_sp->GetFileSpec().GetDirectory().AsCString(),
                            module_sp->GetFileSpec().GetFilename().AsCString(),
                            file, (uint64_t) offset, (uint64_t) length);

        Archive::shared_ptr archive_sp (Archive::FindCachedArchive (*file, module_sp->GetArchitecture(), module_sp->GetModificationTime()));

        std::auto_ptr<ObjectContainerBSDArchive> container_ap(new ObjectContainerBSDArchive (module_sp, data_sp, file, offset, length));

        if (container_ap.get())
        {
            if (archive_sp)
            {
                // We already have this archive in our cache, use it
                container_ap->SetArchive (archive_sp);
                return container_ap.release();
            }
            else if (container_ap->ParseHeader())
                return container_ap.release();
        }
    }
    return NULL;
}



bool
ObjectContainerBSDArchive::MagicBytesMatch (const DataExtractor &data)
{
    uint32_t offset = 0;
    const char* armag = (const char* )data.PeekData (offset, sizeof(ar_hdr));
    if (armag && ::strncmp(armag, ARMAG, SARMAG) == 0)
    {
        armag += offsetof(struct ar_hdr, ar_fmag) + SARMAG;
        if (strncmp(armag, ARFMAG, 2) == 0)
            return true;
    }
    return false;
}

ObjectContainerBSDArchive::ObjectContainerBSDArchive
(
    const lldb::ModuleSP &module_sp,
    DataBufferSP& dataSP,
    const lldb_private::FileSpec *file,
    lldb::addr_t offset,
    lldb::addr_t size
) :
    ObjectContainer (module_sp, file, offset, size, dataSP),
    m_archive_sp ()
{
}
void
ObjectContainerBSDArchive::SetArchive (Archive::shared_ptr &archive_sp)
{
    m_archive_sp  = archive_sp;
}



ObjectContainerBSDArchive::~ObjectContainerBSDArchive()
{
}

bool
ObjectContainerBSDArchive::ParseHeader ()
{
    if (m_archive_sp.get() == NULL)
    {
        if (m_data.GetByteSize() > 0)
        {
            ModuleSP module_sp (GetModule());
            if (module_sp)
            {
                m_archive_sp = Archive::ParseAndCacheArchiveForFile (m_file,
                                                                     module_sp->GetArchitecture(),
                                                                     module_sp->GetModificationTime(),
                                                                     m_data);
            }
        }
    }
    return m_archive_sp.get() != NULL;
}

void
ObjectContainerBSDArchive::Dump (Stream *s) const
{
    s->Printf("%p: ", this);
    s->Indent();
    const size_t num_archs = GetNumArchitectures();
    const size_t num_objects = GetNumObjects();
    s->Printf("ObjectContainerBSDArchive, num_archs = %lu, num_objects = %lu", num_archs, num_objects);
    uint32_t i;
    ArchSpec arch;
    s->IndentMore();
    for (i=0; i<num_archs; i++)
    {
        s->Indent();
        GetArchitectureAtIndex(i, arch);
        s->Printf("arch[%u] = %s\n", i, arch.GetArchitectureName());
    }
    for (i=0; i<num_objects; i++)
    {
        s->Indent();
        s->Printf("object[%u] = %s\n", i, GetObjectNameAtIndex (i));
    }
    s->IndentLess();
    s->EOL();
}

ObjectFileSP
ObjectContainerBSDArchive::GetObjectFile (const FileSpec *file)
{
    ModuleSP module_sp (GetModule());
    if (module_sp)
    {
        if (module_sp->GetObjectName() && m_archive_sp)
        {
            Object *object = m_archive_sp->FindObject (module_sp->GetObjectName());
            if (object)
                return ObjectFile::FindPlugin (module_sp, 
                                               file, 
                                               object->ar_file_offset, 
                                               object->ar_file_size, 
                                               m_data.GetSharedDataBuffer());
        }
    }
    return ObjectFileSP();
}


//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------
const char *
ObjectContainerBSDArchive::GetPluginName()
{
    return "object-container.bsd-archive";
}

const char *
ObjectContainerBSDArchive::GetShortPluginName()
{
    return GetPluginNameStatic();
}

uint32_t
ObjectContainerBSDArchive::GetPluginVersion()
{
    return 1;
}

