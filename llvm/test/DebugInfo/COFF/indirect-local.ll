; RUN: llc < %s | FileCheck %s --check-prefix=ASM
; RUN: llc < %s -filetype=obj | llvm-readobj - --codeview | FileCheck %s --check-prefix=OBJ


; C++ source to regenerate:
; struct Foo {
;   int a;
;   int b;
;   int c;
;   int d;
; };
;
; int main() {
;   Foo f{1, 2, 3, 4};
;   auto &[a, b, c, d] = f;
;   return a;
; }
; $ clang t.cpp -S -emit-llvm -g -o t.ll -std=c++17

; ASM-LABEL:         .long   241                             # Symbol subsection for main
; ASM:               .short  4414                            # Record kind: S_LOCAL
; ASM-NEXT:          .long   4101                            # TypeIndex
; ASM-NEXT:          .short  0                               # Flags
; ASM-NEXT:          .asciz  "f"
; ASM-NEXT:          .p2align        2, 0x0
; ASM-NEXT:  .Ltmp{{.*}}:
; ASM-NEXT:          .cv_def_range    .Ltmp0 .Ltmp1, frame_ptr_rel, 16

; ASM:               .short  4414                            # Record kind: S_LOCAL
; ASM-NEXT:          .long   116                             # TypeIndex
; ASM-NEXT:          .short  0                               # Flags
; ASM-NEXT:          .asciz  "a"
; ASM-NEXT:          .p2align        2, 0x0
; ASM-NEXT:  .Ltmp{{.*}}:
; ASM-NEXT:          .cv_def_range    .Ltmp{{.*}} .Ltmp{{.*}}, reg_rel_indir, 335, 0, 0, 0

; ASM:               .short  4414                            # Record kind: S_LOCAL
; ASM-NEXT:          .long   116                             # TypeIndex
; ASM-NEXT:          .short  0                               # Flags
; ASM-NEXT:          .asciz  "b"
; ASM-NEXT:          .p2align        2, 0x0
; ASM-NEXT:  .Ltmp{{.*}}:
; ASM-NEXT:          .cv_def_range    .Ltmp{{.*}} .Ltmp{{.*}}, reg_rel_indir, 335, 0, 0, 4

; ASM:               .short  4414                            # Record kind: S_LOCAL
; ASM-NEXT:          .long   116                             # TypeIndex
; ASM-NEXT:          .short  0                               # Flags
; ASM-NEXT:          .asciz  "c"
; ASM-NEXT:          .p2align        2, 0x0
; ASM-NEXT:  .Ltmp{{.*}}:
; ASM-NEXT:          .cv_def_range    .Ltmp{{.*}} .Ltmp{{.*}}, reg_rel_indir, 335, 0, 0, 8

; ASM:               .short  4414                            # Record kind: S_LOCAL
; ASM-NEXT:          .long   116                             # TypeIndex
; ASM-NEXT:          .short  0                               # Flags
; ASM-NEXT:          .asciz  "d"
; ASM-NEXT:          .p2align        2, 0x0
; ASM-NEXT:  .Ltmp{{.*}}:
; ASM-NEXT:          .cv_def_range    .Ltmp{{.*}} .Ltmp{{.*}}, reg_rel_indir, 335, 0, 0, 12

; OBJ: Subsection [
; OBJ:   SubSectionType: Symbols (0xF1)
; OBJ:   GlobalProcIdSym {
; OBJ:     FunctionType: main (0x1002)
; OBJ:   }
; OBJ:   LocalSym {
; OBJ:     Kind: S_LOCAL (0x113E)
; OBJ:     Type: Foo (0x1005)
; OBJ:     VarName: f
; OBJ:   }
; OBJ:   DefRangeFramePointerRelSym {
; OBJ:     Kind: S_DEFRANGE_FRAMEPOINTER_REL (0x1142)
; OBJ:     Offset: 16
; OBJ:     LocalVariableAddrRange {
; OBJ:     }
; OBJ:   }
; OBJ:   LocalSym {
; OBJ:     Kind: S_LOCAL (0x113E)
; OBJ:     Type: int (0x74)
; OBJ:     Flags [ (0x0)
; OBJ:     ]
; OBJ:     VarName: a
; OBJ:   }
; OBJ:   DefRangeRegisterRelIndirSym {
; OBJ:     Kind: S_DEFRANGE_REGISTER_REL_INDIR (0x1177)
; OBJ:     BaseRegister: RSP (0x14F)
; OBJ:     HasSpilledUDTMember: No
; OBJ:     OffsetInParent: 0
; OBJ:     BasePointerOffset: 0
; OBJ:     OffsetInUDT: 0
; OBJ:     LocalVariableAddrRange {
; OBJ:     }
; OBJ:   }
; OBJ:   LocalSym {
; OBJ:     Kind: S_LOCAL (0x113E)
; OBJ:     Type: int (0x74)
; OBJ:     Flags [ (0x0)
; OBJ:     ]
; OBJ:     VarName: b
; OBJ:   }
; OBJ:   DefRangeRegisterRelIndirSym {
; OBJ:     Kind: S_DEFRANGE_REGISTER_REL_INDIR (0x1177)
; OBJ:     BaseRegister: RSP (0x14F)
; OBJ:     HasSpilledUDTMember: No
; OBJ:     OffsetInParent: 0
; OBJ:     BasePointerOffset: 0
; OBJ:     OffsetInUDT: 4
; OBJ:     LocalVariableAddrRange {
; OBJ:     }
; OBJ:   }
; OBJ:   LocalSym {
; OBJ:     Kind: S_LOCAL (0x113E)
; OBJ:     Type: int (0x74)
; OBJ:     Flags [ (0x0)
; OBJ:     ]
; OBJ:     VarName: c
; OBJ:   }
; OBJ:   DefRangeRegisterRelIndirSym {
; OBJ:     Kind: S_DEFRANGE_REGISTER_REL_INDIR (0x1177)
; OBJ:     BaseRegister: RSP (0x14F)
; OBJ:     HasSpilledUDTMember: No
; OBJ:     OffsetInParent: 0
; OBJ:     BasePointerOffset: 0
; OBJ:     OffsetInUDT: 8
; OBJ:     LocalVariableAddrRange {
; OBJ:     }
; OBJ:   }
; OBJ:   LocalSym {
; OBJ:     Kind: S_LOCAL (0x113E)
; OBJ:     Type: int (0x74)
; OBJ:     Flags [ (0x0)
; OBJ:     ]
; OBJ:     VarName: d
; OBJ:   }
; OBJ:   DefRangeRegisterRelIndirSym {
; OBJ:     Kind: S_DEFRANGE_REGISTER_REL_INDIR (0x1177)
; OBJ:     BaseRegister: RSP (0x14F)
; OBJ:     HasSpilledUDTMember: No
; OBJ:     OffsetInParent: 0
; OBJ:     BasePointerOffset: 0
; OBJ:     OffsetInUDT: 12
; OBJ:     LocalVariableAddrRange {
; OBJ:     }
; OBJ:   }

; ModuleID = 't.cpp'
source_filename = "t.cpp"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.50.35725"

%struct.Foo = type { i32, i32, i32, i32 }

@__const.main.f = private unnamed_addr constant %struct.Foo { i32 1, i32 2, i32 3, i32 4 }, align 4

; Function Attrs: mustprogress noinline norecurse nounwind optnone uwtable
define dso_local noundef i32 @main() #0 !dbg !9 {
  %1 = alloca i32, align 4
  %2 = alloca %struct.Foo, align 4
  %3 = alloca ptr, align 8
  store i32 0, ptr %1, align 4
    #dbg_declare(ptr %2, !14, !DIExpression(), !21)
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %2, ptr align 4 @__const.main.f, i64 16, i1 false), !dbg !21
    #dbg_declare(ptr %3, !22, !DIExpression(DW_OP_deref), !23)
    #dbg_declare(ptr %3, !24, !DIExpression(DW_OP_deref, DW_OP_plus_uconst, 4), !23)
    #dbg_declare(ptr %3, !25, !DIExpression(DW_OP_deref, DW_OP_plus_uconst, 8), !23)
    #dbg_declare(ptr %3, !26, !DIExpression(DW_OP_deref, DW_OP_plus_uconst, 12), !23)
  store ptr %2, ptr %3, align 8, !dbg !23
  %4 = load ptr, ptr %3, align 8, !dbg !27, !nonnull !13, !align !28
  %5 = getelementptr inbounds nuw %struct.Foo, ptr %4, i32 0, i32 0, !dbg !27
  %6 = load i32, ptr %5, align 4, !dbg !27
  ret i32 %6, !dbg !27
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1

attributes #0 = { mustprogress noinline norecurse nounwind optnone uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 21.1.8", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "t.cpp", directory: "F:\\Dev\\dummy", checksumkind: CSK_MD5, checksum: "e257352cbba404e4548bc4500877ceb0")
!2 = !{i32 2, !"CodeView", i32 1}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 2}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"uwtable", i32 2}
!7 = !{i32 1, !"MaxTLSAlign", i32 65536}
!8 = !{!"clang version 21.1.8"}
!9 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 8, type: !10, scopeLine: 8, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !13)
!10 = !DISubroutineType(types: !11)
!11 = !{!12}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !{}
!14 = !DILocalVariable(name: "f", scope: !9, file: !1, line: 9, type: !15)
!15 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Foo", file: !1, line: 1, size: 128, flags: DIFlagTypePassByValue, elements: !16, identifier: ".?AUFoo@@")
!16 = !{!17, !18, !19, !20}
!17 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !15, file: !1, line: 2, baseType: !12, size: 32)
!18 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !15, file: !1, line: 3, baseType: !12, size: 32, offset: 32)
!19 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !15, file: !1, line: 4, baseType: !12, size: 32, offset: 64)
!20 = !DIDerivedType(tag: DW_TAG_member, name: "d", scope: !15, file: !1, line: 5, baseType: !12, size: 32, offset: 96)
!21 = !DILocation(line: 9, scope: !9)
!22 = !DILocalVariable(name: "a", scope: !9, file: !1, line: 10, type: !12)
!23 = !DILocation(line: 10, scope: !9)
!24 = !DILocalVariable(name: "b", scope: !9, file: !1, line: 10, type: !12)
!25 = !DILocalVariable(name: "c", scope: !9, file: !1, line: 10, type: !12)
!26 = !DILocalVariable(name: "d", scope: !9, file: !1, line: 10, type: !12)
!27 = !DILocation(line: 11, scope: !9)
!28 = !{i64 4}
