; RUN: opt < %s -passes=asan -asan-instrumentation-with-call-threshold=0 -asan-emit-debug-info -S | FileCheck %s --check-prefix=IR
; RUN: opt < %s -passes=asan -asan-instrumentation-with-call-threshold=0 -asan-emit-debug-info -S | llc -mtriple=bpfel -filetype=obj -o %t1
; RUN: llvm-objcopy --dump-section='.BTF'=%t2 %t1
; RUN: %python %p/print_btf.py %t2 | FileCheck %s --check-prefix=ENABLED --implicit-check-not=__asan_store4
; RUN: opt < %s -passes=asan -asan-instrumentation-with-call-threshold=0 -S | llc -mtriple=bpfel -filetype=asm -o - | FileCheck %s --check-prefix=DISABLED --implicit-check-not='.section        .BTF,"",@progbits' --implicit-check-not='.section        .BTF.ext,"",@progbits'

define void @test(ptr %p) sanitize_address {
entry:
  %0 = load i32, ptr %p, align 4
  ret void
}

; ENABLED: [{{[0-9]+}}] FUNC_PROTO '(anon)' ret_type_id=0 vlen=1
; ENABLED-NEXT:     '(anon)' type_id={{[0-9]+}}
; ENABLED: [{{[0-9]+}}] INT '__int_64' size=8 bits_offset=0 nr_bits=64 encoding=SIGNED
; ENABLED: [{{[0-9]+}}] FUNC '__asan_load4' type_id={{[0-9]+}} linkage=extern
; DISABLED: call __asan_load4

; IR: ![[FILE:[0-9]+]] = !DIFile(filename: "asan_interface.h", directory: "sanitizer")
; IR: !DISubprogram(name: "__asan_load4"
; IR-SAME: scope: ![[FILE]]
; IR-SAME: file: ![[FILE]]
; IR-SAME: flags: DIFlagArtificial | DIFlagPrototyped

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "asan-runtime-calls.c", directory: "/tmp")
!2 = !{i32 2, !"Debug Info Version", i32 3}
