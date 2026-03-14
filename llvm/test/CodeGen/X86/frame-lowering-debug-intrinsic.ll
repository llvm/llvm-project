; Test ensuring debug intrinsics do not affect generated function prologue.
;
; RUN: llc -O1 -mtriple=x86_64-unknown-unknown -o - %s | FileCheck %s

define i64 @fn1NoDebug(i64 %a) {
  %call = call i64 @fn(i64 %a, i64 0)
  ret i64 %call
}

; CHECK-LABEL: fn1NoDebug
; CHECK: popq %rcx
; CHECK-NEXT: .cfi_def_cfa_offset 8
; CHECK-NEXT: ret

define i64 @fn1WithDebug(i64 %a) !dbg !4 {
  %call = call i64 @fn(i64 %a, i64 0)
  tail call void @llvm.dbg.value(metadata i64 %call, i64 0, metadata !5, metadata !6), !dbg !7
  ret i64 %call
}

; CHECK-LABEL: fn1WithDebug
; CHECK: popq %rcx
; CHECK-NEXT: .cfi_def_cfa_offset 8
; CHECK-NEXT: ret

%struct.Buffer = type { i8, [63 x i8] }

define void @fn2NoDebug(ptr byval(%struct.Buffer) align 64 %p1) {
  ret void
}

; CHECK-LABEL: fn2NoDebug
; CHECK: and
; CHECK-NOT: add
; CHECK-NOT: sub
; CHECK: mov
; CHECK-NEXT: pop
; CHECK-NEXT: .cfi_def_cfa %rsp, 8
; CHECK-NEXT: ret

define void @fn2WithDebug(ptr byval(%struct.Buffer) align 64 %p1) !dbg !8 {
  call void @llvm.dbg.declare(metadata ptr %p1, metadata !9, metadata !6), !dbg !10
  ret void
}

; CHECK-LABEL: fn2WithDebug
; CHECK: and
; CHECK-NOT: add
; CHECK-NOT: sub
; CHECK: mov
; CHECK-NEXT: pop
; CHECK-NEXT: .cfi_def_cfa %rsp, 8
; CHECK-NEXT: ret

declare i64 @fn(i64, i64)

declare void @llvm.dbg.value(metadata, i64, metadata, metadata)
declare void @llvm.dbg.declare(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2,!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 4.0.0")
!1 = !DIFile(filename: "test.c", directory: "/")
!2 = !{i32 2, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "withDebug", unit: !0)
!5 = !DILocalVariable(name: "w", scope: !4)
!6 = !DIExpression()
!7 = !DILocation(line: 210, column: 12, scope: !4)
!8 = distinct !DISubprogram(name: "withDebug", unit: !0)
!9 = !DILocalVariable(name: "w", scope: !8)
!10 = !DILocation(line: 210, column: 12, scope: !8)
