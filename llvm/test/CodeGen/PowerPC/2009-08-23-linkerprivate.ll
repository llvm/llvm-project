; RUN: llc < %s -march=ppc32 -mtriple=powerpc-apple-darwin | FileCheck %s

; ModuleID = '/Volumes/MacOS9/tests/WebKit/JavaScriptCore/profiler/ProfilerServer.mm'

@"\01l_objc_msgSend_fixup_alloc" = linker_private_weak hidden global i32 0, section "__DATA, __objc_msgrefs, coalesced", align 16

; CHECK: .globl l_objc_msgSend_fixup_alloc
; CHECK: .weak_definition l_objc_msgSend_fixup_alloc
