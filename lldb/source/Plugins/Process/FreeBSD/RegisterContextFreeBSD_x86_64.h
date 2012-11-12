//===-- RegisterContextFreeBSD_x86_64.h ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_RegisterContextFreeBSD_x86_64_H_
#define liblldb_RegisterContextFreeBSD_x86_64_H_

    typedef struct _GPR
    {
        uint64_t r15;
        uint64_t r14;
        uint64_t r13;
        uint64_t r12;
        uint64_t r11;
        uint64_t r10;
        uint64_t r9;
        uint64_t r8;
        uint64_t rdi;
        uint64_t rsi;
        uint64_t rbp;
        uint64_t rbx;
        uint64_t rdx;
        uint64_t rcx;
        uint64_t rax;
        uint32_t trapno;
        uint16_t fs;
        uint16_t gs;
        uint32_t err;
        uint16_t es;
        uint16_t ds;
        uint64_t rip;
        uint64_t cs;
        uint64_t rflags;
        uint64_t rsp;
        uint64_t ss;
    } GPR;

#endif
