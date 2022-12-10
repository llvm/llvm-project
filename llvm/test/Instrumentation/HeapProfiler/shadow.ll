; RUN: opt < %s -passes='function(memprof),module(memprof-module)' -S | FileCheck --check-prefixes=STATIC %s

; RUN: cat %s > %t.pic.ll
; RUN: echo -e '!llvm.module.flags = !{!0}\n!0 = !{i32 7, !"PIC Level", i32 1}' >> %t.pic.ll
; RUN: opt < %t.pic.ll -passes='function(memprof),module(memprof-module)' -S | FileCheck --check-prefixes=PIC %s

; STATIC: @__memprof_shadow_memory_dynamic_address = external dso_local global i64
; PIC: @__memprof_shadow_memory_dynamic_address = external global i64

define i32 @test_load(ptr %a) {
entry:
  %tmp1 = load i32, ptr %a, align 4
  ret i32 %tmp1
}
