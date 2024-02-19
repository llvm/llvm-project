; RUN: opt < %s -passes='mem2reg' -S | FileCheck %s
; RUN: opt < %s -passes='mem2reg' -S --try-experimental-debuginfo-iterators | FileCheck %s
target datalayout = "e-p:64:64"

; An intrinsic without any expressions should always be converted.
define i64 @foo0(i64 %arg) {
  %arg.addr = alloca i64
  store i64 %arg, ptr %arg.addr
  call void @llvm.dbg.declare(metadata ptr %arg.addr, metadata !26, metadata !DIExpression()), !dbg !40
  ; CHECK-LABEL: @foo0
  ; CHECK-SAME:    (i64 [[arg:%.*]])
  ; CHECK-NEXT:    dbg.value(metadata i64 [[arg]], {{.*}}, metadata !DIExpression())
  %val = load i64, ptr %arg.addr
  ret i64 %val
}

; An intrinsic with a single deref operator should be converted preserving the deref.
define i32 @foo1(ptr %arg) {
  %arg.indirect_addr = alloca ptr
  store ptr %arg, ptr %arg.indirect_addr
  call void @llvm.dbg.declare(metadata ptr %arg.indirect_addr, metadata !25, metadata !DIExpression(DW_OP_deref)), !dbg !40
  ; CHECK-LABEL: @foo1
  ; CHECK-SAME:    (ptr [[arg:%.*]])
  ; CHECK-NEXT:    dbg.value(metadata ptr [[arg]], {{.*}}, metadata !DIExpression(DW_OP_deref))
  %val = load i32, ptr %arg
  ret i32 %val
}


; An intrinsic with multiple operators should cause us to conservatively bail.
define i32 @foo2(ptr %arg) {
  %arg.indirect_addr = alloca ptr
  store ptr %arg, ptr %arg.indirect_addr
  call void @llvm.dbg.declare(metadata ptr %arg.indirect_addr, metadata !25, metadata !DIExpression(DW_OP_deref, DW_OP_plus_uconst, 2)), !dbg !40
  ; CHECK-LABEL: @foo2
  ; CHECK-NEXT:     dbg.value(metadata ptr undef, {{.*}}, metadata !DIExpression(DW_OP_deref, DW_OP_plus_uconst, 2))
  %val = load i32, ptr %arg
  ret i32 %val
}

declare void @llvm.dbg.declare(metadata, metadata, metadata)

!llvm.module.flags = !{!1, !2}
!llvm.dbg.cu = !{!7}

!1 = !{i32 7, !"Dwarf Version", i32 4}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DIBasicType(name: "wide type", size: 512)
!4 = !DIBasicType(name: "ptr sized type", size: 64)
!7 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !8, producer: "clang", emissionKind: FullDebug)
!8 = !DIFile(filename: "test.ll", directory: "")
!10 = distinct !DISubprogram(name: "blah", linkageName: "blah", scope: !8, file: !8, line: 7, unit: !7)
!25 = !DILocalVariable(name: "blah", arg: 1, scope: !10, file: !8, line: 7, type:!3)
!26 = !DILocalVariable(name: "ptr sized var", arg: 1, scope: !10, file: !8, line: 7, type:!4)
!40 = !DILocation(line: 7, column: 35, scope: !10)
