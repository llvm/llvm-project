; RUN: llc -dwarf-version=4 -filetype=obj -O0 \
; RUN:     -mtriple=x86_64-unknown-linux-gnu < %s \
; RUN:     | llvm-dwarfdump -v - | FileCheck %s

; Verify that the empty DW_OP_piece operations that are created for the
; optimized-out struct fields `foo' and `baz' are emitted before their
; succeeding fields' symbol location expressions.

; Test file based on foo.c:
;
; struct {
;   long foo;
;   void *bar;
;   long baz;
;   void *qux;
; } static var;
; 
; void *ptr;
; 
; int main() {
;   if (var.foo == 0)
;     var.bar = var.qux = ptr;
;   return 0;
; }
;
; which was built using:
;
; clang -O0 -g2 -S -emit-llvm foo.c -o foo.ll
; opt -S -globalopt foo.ll -o foo.opt.ll

; CHECK: DW_AT_name {{.*}}"var"
; CHECK: DW_AT_location [DW_FORM_exprloc] (DW_OP_piece 0x8, DW_OP_addr 0x0, DW_OP_piece 0x8, DW_OP_piece 0x8, DW_OP_addr 0x0, DW_OP_piece 0x8)

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@var.1 = internal unnamed_addr global ptr null, align 8, !dbg !0
@var.3 = internal unnamed_addr global ptr null, align 8, !dbg !15

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!16, !17}
!llvm.ident = !{!18}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression(DW_OP_LLVM_fragment, 64, 64))
!1 = distinct !DIGlobalVariable(name: "var", scope: !2, file: !3, line: 15, type: !7, isLocal: true, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "foo.c", directory: "/")
!4 = !{}
!5 = !{!6}
!6 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!7 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !3, line: 10, size: 256, elements: !8)
!8 = !{!9, !11, !13, !14}
!9 = !DIDerivedType(tag: DW_TAG_member, name: "foo", scope: !7, file: !3, line: 11, baseType: !10, size: 64)
!10 = !DIBasicType(name: "long int", size: 64, encoding: DW_ATE_signed)
!11 = !DIDerivedType(tag: DW_TAG_member, name: "bar", scope: !7, file: !3, line: 12, baseType: !12, size: 64, offset: 64)
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!13 = !DIDerivedType(tag: DW_TAG_member, name: "baz", scope: !7, file: !3, line: 13, baseType: !10, size: 64, offset: 128)
!14 = !DIDerivedType(tag: DW_TAG_member, name: "qux", scope: !7, file: !3, line: 14, baseType: !12, size: 64, offset: 192)
!15 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression(DW_OP_LLVM_fragment, 192, 64))
!16 = !{i32 2, !"Dwarf Version", i32 4}
!17 = !{i32 2, !"Debug Info Version", i32 3}
!18 = !{!"clang version 7.0.0"}
