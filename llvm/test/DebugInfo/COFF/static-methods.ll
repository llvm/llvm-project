; RUN: llc < %s -filetype=obj | llvm-readobj - -codeview | FileCheck %s

; Check for the appropriate MethodKind below.

; C++ source used to generate IR:
; $ cat t.cpp
; struct A {
;   static void f();
;   void g();
;   static void h(int x, int y);
; };
; A *p = new A;
; $ clang t.cpp -S -emit-llvm -g -gcodeview -o t.ll

; CHECK:       MemberFunction ([[STATIC_VOID:0x.*]]) {
; CHECK-NEXT:    TypeLeafKind: LF_MFUNCTION (0x1009)
; CHECK-NEXT:    ReturnType: void (0x3)
; CHECK-NEXT:    ClassType: A ({{.*}})
; CHECK-NEXT:    ThisType: 0x0
; CHECK-NEXT:    CallingConvention: NearC (0x0)
; CHECK-NEXT:    FunctionOptions [ (0x0)
; CHECK-NEXT:    ]
; CHECK-NEXT:    NumParameters: 0
; CHECK-NEXT:    ArgListType: () ({{.*}})
; CHECK-NEXT:    ThisAdjustment: 0
; CHECK-NEXT:  }
; CHECK:       MemberFunction ([[INSTANCE_VOID:0x.*]]) {
; CHECK-NEXT:    TypeLeafKind: LF_MFUNCTION (0x1009)
; CHECK-NEXT:    ReturnType: void (0x3)
; CHECK-NEXT:    ClassType: A ({{.*}})
; CHECK-NEXT:    ThisType: A* ({{.*}})
; CHECK-NEXT:    CallingConvention: ThisCall (0xB)
; CHECK-NEXT:    FunctionOptions [ (0x0)
; CHECK-NEXT:    ]
; CHECK-NEXT:    NumParameters: 0
; CHECK-NEXT:    ArgListType: () ({{.*}})
; CHECK-NEXT:    ThisAdjustment: 0
; CHECK-NEXT:  }
; CHECK:       MemberFunction ([[STATIC_TWO:0x.*]]) {
; CHECK-NEXT:    TypeLeafKind: LF_MFUNCTION (0x1009)
; CHECK-NEXT:    ReturnType: void (0x3)
; CHECK-NEXT:    ClassType: A ({{.*}})
; CHECK-NEXT:    ThisType: 0x0
; CHECK-NEXT:    CallingConvention: NearC (0x0)
; CHECK-NEXT:    FunctionOptions [ (0x0)
; CHECK-NEXT:    ]
; CHECK-NEXT:    NumParameters: 2
; CHECK-NEXT:    ArgListType: (int, int) ({{.*}}
; CHECK-NEXT:    ThisAdjustment: 0
; CHECK-NEXT:  }
; CHECK:      OneMethod {
; CHECK-NEXT:   TypeLeafKind: LF_ONEMETHOD (0x1511)
; CHECK-NEXT:   AccessSpecifier: Public (0x3)
; CHECK-NEXT:   MethodKind: Static (0x2)
; CHECK-NEXT:   Type: void A::() ([[STATIC_VOID]])
; CHECK-NEXT:   Name: f
; CHECK-NEXT: }
; CHECK-NEXT: OneMethod {
; CHECK-NEXT:   TypeLeafKind: LF_ONEMETHOD (0x1511)
; CHECK-NEXT:   AccessSpecifier: Public (0x3)
; CHECK-NEXT:   Type: void A::() ([[INSTANCE_VOID]])
; CHECK-NEXT:   Name: g
; CHECK-NEXT: }
; CHECK-NEXT: OneMethod {
; CHECK-NEXT:   TypeLeafKind: LF_ONEMETHOD (0x1511)
; CHECK-NEXT:   AccessSpecifier: Public (0x3)
; CHECK-NEXT:   MethodKind: Static (0x2)
; CHECK-NEXT:   Type: void A::(int, int) ([[STATIC_TWO]])
; CHECK-NEXT:   Name: h
; CHECK-NEXT: }

; ModuleID = 't.cpp'
source_filename = "t.cpp"
target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc19.0.24215"

%struct.A = type { i8 }

@"\01?p@@3PAUA@@A" = global %struct.A* null, align 4, !dbg !0
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_t.cpp, i8* null }]

; Function Attrs: noinline
define internal void @"\01??__Ep@@YAXXZ"() #0 !dbg !25 {
entry:
  %call = call i8* @"\01??2@YAPAXI@Z"(i32 1) #2, !dbg !26
  %0 = bitcast i8* %call to %struct.A*, !dbg !26
  store %struct.A* %0, %struct.A** @"\01?p@@3PAUA@@A", align 4, !dbg !26
  ret void, !dbg !27
}

; Function Attrs: nobuiltin
declare noalias i8* @"\01??2@YAPAXI@Z"(i32) #1

; Function Attrs: noinline
define internal void @_GLOBAL__sub_I_t.cpp() #0 !dbg !28 {
entry:
  call void @"\01??__Ep@@YAXXZ"(), !dbg !30
  ret void
}

attributes #0 = { noinline "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nobuiltin "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { builtin }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!20, !21, !22, !23}
!llvm.ident = !{!24}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "p", linkageName: "\01?p@@3PAUA@@A", scope: !2, file: !3, line: 7, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 6.0.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "t.cpp", directory: "D:\5Csrc\5Ct", checksumkind: CSK_MD5, checksum: "168f4e5caded7d526f64a37783785c64")
!4 = !{}
!5 = !{!0}
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 32)
!7 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "A", file: !3, line: 1, size: 8, elements: !8, identifier: ".?AUA@@")
!8 = !{!9, !12, !16}
!9 = !DISubprogram(name: "f", linkageName: "\01?f@A@@SAXXZ", scope: !7, file: !3, line: 2, type: !10, isLocal: false, isDefinition: false, scopeLine: 2, flags: DIFlagPrototyped | DIFlagStaticMember, isOptimized: false)
!10 = !DISubroutineType(types: !11)
!11 = !{null}
!12 = !DISubprogram(name: "g", linkageName: "\01?g@A@@QAEXXZ", scope: !7, file: !3, line: 3, type: !13, isLocal: false, isDefinition: false, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false)
!13 = !DISubroutineType(cc: DW_CC_BORLAND_thiscall, types: !14)
!14 = !{null, !15}
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 32, flags: DIFlagArtificial | DIFlagObjectPointer)
!16 = !DISubprogram(name: "h", linkageName: "\01?h@A@@SAXHH@Z", scope: !7, file: !3, line: 4, type: !17, isLocal: false, isDefinition: false, scopeLine: 4, flags: DIFlagPrototyped | DIFlagStaticMember, isOptimized: false)
!17 = !DISubroutineType(types: !18)
!18 = !{null, !19, !19}
!19 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!20 = !{i32 1, !"NumRegisterParameters", i32 0}
!21 = !{i32 2, !"CodeView", i32 1}
!22 = !{i32 2, !"Debug Info Version", i32 3}
!23 = !{i32 1, !"wchar_size", i32 2}
!24 = !{!"clang version 6.0.0 "}
!25 = distinct !DISubprogram(name: "??__Ep@@YAXXZ", scope: !3, file: !3, line: 7, type: !10, isLocal: true, isDefinition: true, scopeLine: 7, flags: DIFlagPrototyped, isOptimized: false, unit: !2, variables: !4)
!26 = !DILocation(line: 7, column: 8, scope: !25)
!27 = !DILocation(line: 7, column: 12, scope: !25)
!28 = distinct !DISubprogram(linkageName: "_GLOBAL__sub_I_t.cpp", scope: !3, file: !3, type: !29, isLocal: true, isDefinition: true, flags: DIFlagArtificial, isOptimized: false, unit: !2, variables: !4)
!29 = !DISubroutineType(types: !4)
!30 = !DILocation(line: 0, scope: !28)
