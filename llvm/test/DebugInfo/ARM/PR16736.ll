; RUN: llc -filetype=asm < %s | FileCheck %s
; RUN: llc -filetype=obj < %s \
; RUN:   | llvm-dwarfdump -debug-info - | FileCheck %s --check-prefix=DWARF
;
; CHECK: @DEBUG_VALUE: h:x <- [DW_OP_plus_uconst {{.*}}] [%r{{.*}}+0]
; DWARF: DW_TAG_formal_parameter
; DWARF:       DW_AT_location
; DWARF-NEXT:    DW_OP_reg0 R0
; DWARF: DW_TAG_formal_parameter
; DWARF:       DW_AT_location
; DWARF-NEXT:    DW_OP_reg1 R1
; DWARF: DW_TAG_formal_parameter
; DWARF:       DW_AT_location
; DWARF-NEXT:    DW_OP_reg2 R2
; DWARF: DW_TAG_formal_parameter
; DWARF:       DW_AT_location
; DWARF-NEXT:    DW_OP_reg3 R3
; DWARF: DW_TAG_formal_parameter
; DWARF: DW_AT_location
; DWARF-NEXT: DW_OP_breg7 R7+8
; generated from:
; clang -cc1 -triple  thumbv7 -S -O1 arm.cpp  -g
;
; int f();
; void g(float);
; void h(int, int, int, int, float x) {
;    g(x = f());
; }
;
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:64:128-a0:0:32-n32-S64"
target triple = "thumbv7-apple-ios"

; Function Attrs: nounwind
define arm_aapcscc void @_Z1hiiiif(i32, i32, i32, i32, float %x) #0 "no-frame-pointer-elim"="true" !dbg !4 {
entry:
  tail call void @llvm.dbg.value(metadata i32 %0, metadata !12, metadata !DIExpression()), !dbg !18
  tail call void @llvm.dbg.value(metadata i32 %1, metadata !13, metadata !DIExpression()), !dbg !18
  tail call void @llvm.dbg.value(metadata i32 %2, metadata !14, metadata !DIExpression()), !dbg !18
  tail call void @llvm.dbg.value(metadata i32 %3, metadata !15, metadata !DIExpression()), !dbg !18
  tail call void @llvm.dbg.value(metadata float %x, metadata !16, metadata !DIExpression()), !dbg !18
  %call = tail call arm_aapcscc i32 @_Z1fv() #3, !dbg !19
  %conv = sitofp i32 %call to float, !dbg !19
  tail call void @llvm.dbg.value(metadata float %conv, metadata !16, metadata !DIExpression()), !dbg !19
  tail call arm_aapcscc void @_Z1gf(float %conv) #3, !dbg !19
  ret void, !dbg !20
}

declare arm_aapcscc void @_Z1gf(float)

declare arm_aapcscc i32 @_Z1fv()

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { nounwind  }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!17, !21}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.4 (trunk 190804) (llvm/trunk 190797)", isOptimized: true, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "/<unknown>", directory: "")
!2 = !{}
!4 = distinct !DISubprogram(name: "h", linkageName: "_Z1hiiiif", line: 3, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !0, scopeLine: 3, file: !5, scope: !6, type: !7, variables: !11)
!5 = !DIFile(filename: "/arm.cpp", directory: "")
!6 = !DIFile(filename: "/arm.cpp", directory: "")
!7 = !DISubroutineType(types: !8)
!8 = !{null, !9, !9, !9, !9, !10}
!9 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DIBasicType(tag: DW_TAG_base_type, name: "float", size: 32, align: 32, encoding: DW_ATE_float)
!11 = !{!12, !13, !14, !15, !16}
!12 = !DILocalVariable(name: "", line: 3, arg: 1, scope: !4, file: !6, type: !9)
!13 = !DILocalVariable(name: "", line: 3, arg: 2, scope: !4, file: !6, type: !9)
!14 = !DILocalVariable(name: "", line: 3, arg: 3, scope: !4, file: !6, type: !9)
!15 = !DILocalVariable(name: "", line: 3, arg: 4, scope: !4, file: !6, type: !9)
!16 = !DILocalVariable(name: "x", line: 3, arg: 5, scope: !4, file: !6, type: !10)
!17 = !{i32 2, !"Dwarf Version", i32 4}
!18 = !DILocation(line: 3, scope: !4)
!19 = !DILocation(line: 4, scope: !4)
!20 = !DILocation(line: 5, scope: !4)
!21 = !{i32 1, !"Debug Info Version", i32 3}
