//===-- MachDYLD.h ----------------------------------------------*- C++ -*-===//
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

#ifndef __MachDYLD_h__
#define __MachDYLD_h__

#include "DNBDefs.h"
#include "DNBRuntimeAction.h"
#include "PThreadMutex.h"
#include <map>
#include <vector>
#include <string>

class DNBBreakpoint;
class MachProcess;

class MachDYLD : public DNBRuntimeAction
{
public:
    MachDYLD ();
    virtual ~MachDYLD ();

    //------------------------------------------------------------------
    // DNBRuntimeAction required functions
    //------------------------------------------------------------------
    virtual void Initialize(nub_process_t pid);
    virtual void ProcessStateChanged(nub_state_t state);
    virtual void SharedLibraryStateChanged(DNBExecutableImageInfo *image_infos, nub_size_t num_image_infos);

protected:
    bool                CheckForDYLDInMemory();
    bool                FoundDYLD() const;
    void                Clear();
    void                Dump(FILE *f) const;
    nub_process_t       ProcessID() const { return m_pid; }
    uint32_t            AddrByteSize() const { return m_addr_size; }
    nub_size_t          CopyCurrentShlibInfo(DNBExecutableImageInfo **image_infos);
    nub_size_t          CopyChangedShlibInfo(DNBExecutableImageInfo **image_infos);
    nub_addr_t          GetSharedLibraryHeaderAddress(const char *shlib_path) const;
    bool                CheckForDYLDInMemory(nub_addr_t addr);
    bool                ReadDYLIBInfo ();
    static nub_bool_t   BreakpointHit (nub_process_t pid, nub_thread_t tid, nub_break_t breakID, void *baton);
    void                UpdateUUIDs();

    struct DYLIBInfo
    {
        nub_addr_t      address;    // Address of mach header for this dylib
        nub_addr_t      mod_date;    // Modification date for this dylib
        std::string     path;        // Resolved path for this dylib
        uint8_t         uuid[16];    // UUID for this dylib if it has one, else all zeros
        std::vector<DNBSegment> segments;    // All segment vmaddr and vmsize pairs for this executable (from memory of inferior)

        DYLIBInfo() :
            address(INVALID_NUB_ADDRESS),
            mod_date(0),
            path(),
            segments()
        {
            memset(uuid, 0, 16);
        }

        void Clear()
        {
            address = INVALID_NUB_ADDRESS;
            mod_date = 0;
            path.clear();
            segments.clear();
            memset(uuid, 0, 16);
        }

        bool operator == (const DYLIBInfo& rhs) const
        {
            return    address        == rhs.address
                 && mod_date    == rhs.mod_date
                 && path        == rhs.path
                 && memcmp(uuid, rhs.uuid, 16) == 0;
        }
        bool UUIDValid() const
        {
            return  uuid[ 0] || uuid[ 1] || uuid[ 2] || uuid[ 3] ||
                    uuid[ 4] || uuid[ 5] || uuid[ 6] || uuid[ 7] ||
                    uuid[ 8] || uuid[ 9] || uuid[10] || uuid[11] ||
                    uuid[12] || uuid[13] || uuid[14] || uuid[15];
        }

        void Dump(FILE *f) const;
        typedef std::vector<DYLIBInfo> collection;
        typedef collection::iterator iterator;
        typedef collection::const_iterator const_iterator;
    };
    struct InfoHeader
    {
        uint32_t    version;        /* == 1 in Mac OS X 10.4, == 2 in Mac OS 10.5 */
        uint32_t    dylib_info_count;
        nub_addr_t  dylib_info_addr;
        nub_addr_t  notification;
        bool        processDetachedFromSharedRegion;

        InfoHeader() :
            version(0),
            dylib_info_count(0),
            dylib_info_addr(INVALID_NUB_ADDRESS),
            notification(INVALID_NUB_ADDRESS),
            processDetachedFromSharedRegion(false)
        {
        }

        void Clear()
        {
            version = 0;
            dylib_info_count = 0;
            dylib_info_addr = INVALID_NUB_ADDRESS;
            notification = INVALID_NUB_ADDRESS;
            processDetachedFromSharedRegion = false;
        }

        bool IsValid() const
        {
            return version == 1 || version == 2;
        }
    };
    static nub_size_t   CopySharedLibraryInfo(DYLIBInfo::collection& dylib_coll, DNBExecutableImageInfo **image_infos);
    static nub_size_t   CopySharedInfoCallback(nub_process_t pid, struct DNBExecutableImageInfo **image_infos, nub_bool_t only_changed, void *baton);
    nub_process_t       m_pid;
    uint32_t            m_addr_size;
    nub_addr_t              m_dyld_addr;
    nub_addr_t              m_dyld_all_image_infos_addr;
    InfoHeader              m_dylib_info_header;
    DYLIBInfo::collection   m_current_dylibs;    // Current shared libraries information
    DYLIBInfo::collection   m_changed_dylibs;    // Shared libraries that changed since last shared library update
    nub_break_t             m_notify_break_id;
    mutable PThreadMutex    m_dyld_info_mutex;
};

#endif // #ifndef __MachDYLD_h__
