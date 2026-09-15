; RUN: llc -mtriple=i386-windows-msvc -filetype=asm < %s | FileCheck %s --check-prefixes=CHECK,ATT
; RUN: llc -mtriple=i386-windows-msvc -x86-asm-syntax=intel -filetype=asm < %s | FileCheck %s --check-prefixes=CHECK,INTEL
; RUN: llc -mtriple=i386-windows-msvc -filetype=obj < %s | llvm-readobj --codeview - | FileCheck %s --check-prefix=OBJ

; Check that object emission puts each location at the corresponding encoded
; instruction, not merely at successive byte addresses.
; OBJ: FunctionLineTable [
; OBJ-NEXT: Name: _gnu
; OBJ: LineNumberStart: 5
; OBJ: +0x[[#%x,START:]] [
; OBJ-NEXT: LineNumberStart: 10
; OBJ: +0x[[#%x,START + 2]] [
; OBJ-NEXT: LineNumberStart: 11
; OBJ: +0x[[#%x,START + 5]] [
; OBJ-NEXT: LineNumberStart: 12

; Substituting $0 grows the string, while unescaping $$ shrinks it. The
; instructions share an asm line but originate on different source lines.
; CHECK-LABEL: _gnu:
; CHECK: #APP
; CHECK: .cv_loc 0 1 10 3
; ATT-NEXT: movl %ecx, %eax
; INTEL-NEXT: mov eax, ecx
; CHECK-NEXT: .cv_loc 0 1 11 5
; ATT-NEXT: addl $1, %eax
; INTEL-NEXT: add eax, 1
; CHECK-NEXT: .cv_loc 0 1 12 7
; ATT-NEXT: incl %eax
; INTEL-NEXT: inc eax
; CHECK: #NO_APP

define void @gnu(i32 %x) !dbg !8 {
  call void asm sideeffect "movl $0, %eax;addl $$1, %eax;incl %eax", "{ecx},~{eax},~{flags}"(i32 %x), !srcloc !12, !dbg !13
  ret void, !dbg !13
}

; Both dialect alternatives have locations. Test selecting either alternative
; and the common instruction that follows it, including the Intel prefix.
; CHECK-LABEL: _variants:
; CHECK: #APP
; CHECK: .cv_loc 1 1 21 3
; ATT-NEXT: incl %eax
; INTEL-NEXT: inc eax
; CHECK-NEXT: .cv_loc 1 1 23 3
; CHECK-NEXT: nop
; CHECK: #NO_APP
; CHECK: #APP
; CHECK: .cv_loc 1 1 22 3
; ATT-NEXT: incl %eax
; INTEL-NEXT: inc eax
; CHECK-NEXT: .cv_loc 1 1 23 3
; CHECK-NEXT: nop
; CHECK: #NO_APP

define void @variants() !dbg !20 {
  call void asm sideeffect "$(incl %eax$|inc eax$); nop", "~{eax},~{flags}"(), !srcloc !22, !dbg !21
  call void asm sideeffect inteldialect "$(incl %eax$|inc eax$); nop", "~{eax},~{flags}"(), !srcloc !22, !dbg !21
  ret void, !dbg !21
}

; An invalid (unsorted) location table must not suppress the ordinary !dbg
; location or crash while looking up offsets.
; CHECK-LABEL: _fallback:
; CHECK: .cv_loc 2 1 30 3
; CHECK: #APP
; CHECK-NOT: .cv_loc
; CHECK: nop
; CHECK-NOT: .cv_loc
; CHECK: #NO_APP

define void @fallback() !dbg !30 {
  call void asm sideeffect "nop", ""(), !srcloc !32, !dbg !31
  ret void, !dbg !31
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "inline-asm-codeview.c", directory: "/")
!2 = !{}
!3 = !{i32 2, !"CodeView", i32 1}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!8 = distinct !DISubprogram(name: "gnu", scope: !1, file: !1, line: 1, type: !5, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!12 = !{i64 1234, !14}
!13 = !DILocation(line: 5, column: 3, scope: !8)
!14 = !{!"inlineasm.dbg.offset", i32 0, i32 10, i32 3, i32 14, i32 11, i32 5, i32 29, i32 12, i32 7}
!20 = distinct !DISubprogram(name: "variants", scope: !1, file: !1, line: 20, type: !5, scopeLine: 20, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!21 = !DILocation(line: 20, column: 3, scope: !20)
!22 = !{i64 1234, !23}
!23 = !{!"inlineasm.dbg.offset", i32 2, i32 21, i32 3, i32 13, i32 22, i32 3, i32 24, i32 23, i32 3}
!30 = distinct !DISubprogram(name: "fallback", scope: !1, file: !1, line: 30, type: !5, scopeLine: 30, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!31 = !DILocation(line: 30, column: 3, scope: !30)
!32 = !{i64 1234, !33}
!33 = !{!"inlineasm.dbg.offset", i32 1, i32 31, i32 3, i32 0, i32 32, i32 3}
