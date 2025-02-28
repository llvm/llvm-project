//===-- AIXCore.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Notes about AIX Process core dumps:
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_AIX_CORE_AIXCORE_H
#define LLDB_SOURCE_PLUGINS_PROCESS_AIX_CORE_AIXCORE_H

#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <cstring>
#include <type_traits>

#include <sys/types.h>
#include <procinfo.h>

namespace AIXCORE {

struct RegContext {
    // The data is arranged in order as filled by AIXCore.cpp in this coredump file
    // so we have to fetch in that exact order, refer there. 
    // But need to change
    // the context structure in order according to Infos_ppc64
        uint64_t                gpr[32];    /* 64-bit gprs */
        unsigned long           pc;            /* msr */
        unsigned long           msr;            /* iar */
        unsigned long           origr3;            /* iar */
        unsigned long           ctr;            /* CTR */
        unsigned long           lr;             /* LR */
        unsigned long           xer;            /* XER */
        unsigned long           cr;             /* CR */
        unsigned long           softe;             /* CR */
        unsigned long           trap;             /* CR */
        unsigned int            fpscr;          /* floating pt status reg */
        unsigned int            fpscrx;         /* software ext to fpscr */
        unsigned long           except[1];      /* exception address    */
        double                  fpr[32];    /* floating pt regs     */
        char                    fpeu;           /* floating pt ever used */
        char                    fpinfo;         /* floating pt info     */
        char                    fpscr24_31;     /* bits 24-31 of 64-bit FPSCR */
        char                    pad[1];
        int                     excp_type;      /* exception type       */
};

    struct ThreadContext64 {
        struct thrdentry64 thread;
        struct RegContext context;
    };

    struct UserData {

        struct procentry64 process;
        unsigned long long reserved[16];
    };

    struct AIXCore64Header {

        int8_t   SignalNum;     /* signal number (cause of error) */    
        int8_t   Flag;      /* flag to describe core dump type */   
        uint16_t Entries;   /* number of core dump modules */           
        uint32_t Version;   /* core file format number */           
        uint64_t FDInfo;  /* offset to fd region in file */

        uint64_t LoaderOffset;    /* offset to loader region in file */
        uint64_t LoaderSize;     /* size of loader region */

        uint32_t NumberOfThreads ;     /* number of elements in thread table */
        uint32_t Reserved0; /* Padding                            */
        uint64_t ThreadContextOffset;       /* offset to thread context table */

        uint64_t NumSegRegion;      /* n of elements in segregion */
        uint64_t SegRegionOffset; /* offset to start of segregion table */

        uint64_t StackOffset;     /* offset of user stack in file */
        uint64_t StackBaseAddr;  /* base address of user stack region */
        uint64_t StackSize;      /* size of user stack region */

        uint64_t DataRegionOffset;      /* offset to user data region */
        uint64_t DataBaseAddr;   /* base address of user data region */
        uint64_t DataSize;  /* size of user data region */
        uint64_t SDataBase;     /* base address of sdata region */
        uint64_t SDataSize;    /* size of sdata region */

        uint64_t NumVMRegions; /* number of anonymously mapped areas */
        uint64_t VMOffset;       /* offset to start of vm_infox table */

        int32_t  ProcessorImplementation;      /* processor implementation */
        uint32_t NumElementsCTX;  /* n of elements in extended ctx table*/
        uint64_t CPRSOffset;      /* Checkpoint/Restart offset */
        uint64_t ExtendedContextOffset;    /* extended context offset */
        uint64_t OffsetUserKey;   /* Offset to user-key exception data */
        uint64_t OffsetLoaderTLS;   /* offset to the loader region in file
                                 when a process uses TLS data */
        uint64_t TLSLoaderSize;    /* size of the above loader region */
        uint64_t ExtendedProcEntry;   /* Extended procentry64 information */
        uint64_t Reserved[2];

        struct ThreadContext64 Fault;

        struct UserData User;

        AIXCore64Header();

        bool ParseCoreHeader(lldb_private::DataExtractor &data,
                lldb::offset_t *offset);
        bool ParseThreadContext(lldb_private::DataExtractor &data,
                lldb::offset_t *offset);
        bool ParseUserData(lldb_private::DataExtractor &data,
                lldb::offset_t *offset);
        bool ParseRegisterContext(lldb_private::DataExtractor &data,
                lldb::offset_t *offset);
        bool ParseLoaderData(lldb_private::DataExtractor &data,
                lldb::offset_t *offset);

    };


}

#endif // LLDB_SOURCE_PLUGINS_PROCESS_AIX_CORE_AIXCORE_H
