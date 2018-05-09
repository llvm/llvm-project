; Checks that llvm.dbg.declare -> llvm.dbg.value conversion utility
; (here exposed through the SROA) pass, properly inserts bit_piece expressions
; if it only describes part of the variable.
; RUN: opt -S -sroa %s | FileCheck %s

; Built from:
; struct foo { bool b; long i; };
; void f(bool b, bool expr, foo g) {
; }
; And modifying the frag dbg.declare to use a fragmented DIExpression (with offset: 0, size: 4)
; to test the dbg.declare+fragment case here.

; Expect two fragments:
; * first starting at bit 0, 8 bits (for the bool)
; * second starting at bit 32, 32 bits (for the long)
; (this happens to create/demonstrate a gap from bits [7, 32))

; But also check that a complex expression is not used for a lone bool
; parameter. It can reference the register it's in directly without masking off
; high bits or anything

; CHECK: call void @llvm.dbg.value(metadata i8 %g.coerce0, metadata ![[VAR_STRUCT:[0-9]+]], metadata !DIExpression(DW_OP_LLVM_fragment, 0, 8))
; CHECK: call void @llvm.dbg.value(metadata i64 %g.coerce1, metadata ![[VAR_STRUCT]], metadata !DIExpression(DW_OP_LLVM_fragment, 32, 64))
; CHECK: call void @llvm.dbg.value(metadata i1 %b, metadata ![[VAR_BOOL:[0-9]+]], metadata !DIExpression())
; CHECK: call void @llvm.dbg.value(metadata i1 %frag, metadata ![[VAR_FRAG:[0-9]+]], metadata !DIExpression(DW_OP_LLVM_fragment, 0, 1))

%struct.foo = type { i8, i64 }

; Function Attrs: noinline nounwind uwtable
define void @_Z1fbb3foo(i1 zeroext %b, i1 zeroext %frag, i8 %g.coerce0, i64 %g.coerce1) #0 !dbg !6 {
entry:
  %g = alloca %struct.foo, align 8
  %b.addr = alloca i8, align 1
  %frag.addr = alloca i8, align 1
  %0 = bitcast %struct.foo* %g to { i8, i64 }*
  %1 = getelementptr inbounds { i8, i64 }, { i8, i64 }* %0, i32 0, i32 0
  store i8 %g.coerce0, i8* %1, align 8
  %2 = getelementptr inbounds { i8, i64 }, { i8, i64 }* %0, i32 0, i32 1
  store i64 %g.coerce1, i64* %2, align 8
  %frombool = zext i1 %b to i8
  store i8 %frombool, i8* %b.addr, align 1
  call void @llvm.dbg.declare(metadata i8* %b.addr, metadata !15, metadata !16), !dbg !17
  %frombool1 = zext i1 %frag to i8
  store i8 %frombool1, i8* %frag.addr, align 1
  call void @llvm.dbg.declare(metadata i8* %frag.addr, metadata !18, metadata !23), !dbg !19
  call void @llvm.dbg.declare(metadata %struct.foo* %g, metadata !20, metadata !16), !dbg !21
  ret void, !dbg !22
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 5.0.0 (trunk 303077) (llvm/trunk 303098)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "foo.cpp", directory: "/usr/local/google/home/blaikie/dev/scratch")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"clang version 5.0.0 (trunk 303077) (llvm/trunk 303098)"}
!6 = distinct !DISubprogram(name: "f", linkageName: "_Z1fbb3foo", scope: !1, file: !1, line: 2, type: !7, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !8)
!8 = !{null, !9, !9, !10}
!9 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!10 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "foo", file: !1, line: 1, size: 128, elements: !11, identifier: "_ZTS3foo")
!11 = !{!12, !13}
!12 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !10, file: !1, line: 1, baseType: !9, size: 8)
!13 = !DIDerivedType(tag: DW_TAG_member, name: "i", scope: !10, file: !1, line: 1, baseType: !14, size: 64, offset: 64)
!14 = !DIBasicType(name: "long int", size: 64, encoding: DW_ATE_signed)
!15 = !DILocalVariable(name: "b", arg: 1, scope: !6, file: !1, line: 2, type: !9)
!16 = !DIExpression()
!17 = !DILocation(line: 2, column: 13, scope: !6)
!18 = !DILocalVariable(name: "frag", arg: 2, scope: !6, file: !1, line: 2, type: !9)
!19 = !DILocation(line: 2, column: 21, scope: !6)
!20 = !DILocalVariable(name: "g", arg: 3, scope: !6, file: !1, line: 2, type: !10)
!21 = !DILocation(line: 2, column: 31, scope: !6)
!22 = !DILocation(line: 3, column: 1, scope: !6)
!23 = !DIExpression(DW_OP_LLVM_fragment, 0, 4)
