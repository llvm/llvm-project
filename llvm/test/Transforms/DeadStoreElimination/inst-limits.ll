; RUN: opt -S -dse < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; If there are two stores to the same location, DSE should be able to remove
; the first store if the two stores are separated by no more than 98
; instructions. The existence of debug intrinsics between the stores should
; not affect this instruction limit.

@x = global i32 0, align 4

; Function Attrs: nounwind
define i32 @test_within_limit() !dbg !4 {
entry:
  ; The first store; later there is a second store to the same location,
  ; so this store should be optimized away by DSE.
  ; CHECK-NOT: store i32 1, i32* @x, align 4
  store i32 1, i32* @x, align 4

  ; Insert 98 dummy instructions between the two stores
  %0 = bitcast i32 0 to i32
  %1 = bitcast i32 0 to i32
  %2 = bitcast i32 0 to i32
  %3 = bitcast i32 0 to i32
  %4 = bitcast i32 0 to i32
  %5 = bitcast i32 0 to i32
  %6 = bitcast i32 0 to i32
  %7 = bitcast i32 0 to i32
  %8 = bitcast i32 0 to i32
  %9 = bitcast i32 0 to i32
  %10 = bitcast i32 0 to i32
  %11 = bitcast i32 0 to i32
  %12 = bitcast i32 0 to i32
  %13 = bitcast i32 0 to i32
  %14 = bitcast i32 0 to i32
  %15 = bitcast i32 0 to i32
  %16 = bitcast i32 0 to i32
  %17 = bitcast i32 0 to i32
  %18 = bitcast i32 0 to i32
  %19 = bitcast i32 0 to i32
  %20 = bitcast i32 0 to i32
  %21 = bitcast i32 0 to i32
  %22 = bitcast i32 0 to i32
  %23 = bitcast i32 0 to i32
  %24 = bitcast i32 0 to i32
  %25 = bitcast i32 0 to i32
  %26 = bitcast i32 0 to i32
  %27 = bitcast i32 0 to i32
  %28 = bitcast i32 0 to i32
  %29 = bitcast i32 0 to i32
  %30 = bitcast i32 0 to i32
  %31 = bitcast i32 0 to i32
  %32 = bitcast i32 0 to i32
  %33 = bitcast i32 0 to i32
  %34 = bitcast i32 0 to i32
  %35 = bitcast i32 0 to i32
  %36 = bitcast i32 0 to i32
  %37 = bitcast i32 0 to i32
  %38 = bitcast i32 0 to i32
  %39 = bitcast i32 0 to i32
  %40 = bitcast i32 0 to i32
  %41 = bitcast i32 0 to i32
  %42 = bitcast i32 0 to i32
  %43 = bitcast i32 0 to i32
  %44 = bitcast i32 0 to i32
  %45 = bitcast i32 0 to i32
  %46 = bitcast i32 0 to i32
  %47 = bitcast i32 0 to i32
  %48 = bitcast i32 0 to i32
  %49 = bitcast i32 0 to i32
  %50 = bitcast i32 0 to i32
  %51 = bitcast i32 0 to i32
  %52 = bitcast i32 0 to i32
  %53 = bitcast i32 0 to i32
  %54 = bitcast i32 0 to i32
  %55 = bitcast i32 0 to i32
  %56 = bitcast i32 0 to i32
  %57 = bitcast i32 0 to i32
  %58 = bitcast i32 0 to i32
  %59 = bitcast i32 0 to i32
  %60 = bitcast i32 0 to i32
  %61 = bitcast i32 0 to i32
  %62 = bitcast i32 0 to i32
  %63 = bitcast i32 0 to i32
  %64 = bitcast i32 0 to i32
  %65 = bitcast i32 0 to i32
  %66 = bitcast i32 0 to i32
  %67 = bitcast i32 0 to i32
  %68 = bitcast i32 0 to i32
  %69 = bitcast i32 0 to i32
  %70 = bitcast i32 0 to i32
  %71 = bitcast i32 0 to i32
  %72 = bitcast i32 0 to i32
  %73 = bitcast i32 0 to i32
  %74 = bitcast i32 0 to i32
  %75 = bitcast i32 0 to i32
  %76 = bitcast i32 0 to i32
  %77 = bitcast i32 0 to i32
  %78 = bitcast i32 0 to i32
  %79 = bitcast i32 0 to i32
  %80 = bitcast i32 0 to i32
  %81 = bitcast i32 0 to i32
  %82 = bitcast i32 0 to i32
  %83 = bitcast i32 0 to i32
  %84 = bitcast i32 0 to i32
  %85 = bitcast i32 0 to i32
  %86 = bitcast i32 0 to i32
  %87 = bitcast i32 0 to i32
  %88 = bitcast i32 0 to i32
  %89 = bitcast i32 0 to i32
  %90 = bitcast i32 0 to i32
  %91 = bitcast i32 0 to i32
  %92 = bitcast i32 0 to i32
  %93 = bitcast i32 0 to i32
  %94 = bitcast i32 0 to i32
  %95 = bitcast i32 0 to i32
  %96 = bitcast i32 0 to i32
  %97 = bitcast i32 0 to i32

  ; Insert a meaningless dbg.value intrinsic; it should have no
  ; effect on the working of DSE in any way.
  call void @llvm.dbg.value(metadata i32 undef, i64 0, metadata !10, metadata !DIExpression()), !dbg !DILocation(scope: !4)

  ; CHECK:  store i32 -1, i32* @x, align 4
  store i32 -1, i32* @x, align 4
  ret i32 0
}

; Function Attrs: nounwind
define i32 @test_outside_limit() {
entry:
  ; The first store; later there is a second store to the same location
  ; CHECK: store i32 1, i32* @x, align 4
  store i32 1, i32* @x, align 4

  ; Insert 99 dummy instructions between the two stores; this is
  ; one too many instruction for the DSE to take place.
  %0 = bitcast i32 0 to i32
  %1 = bitcast i32 0 to i32
  %2 = bitcast i32 0 to i32
  %3 = bitcast i32 0 to i32
  %4 = bitcast i32 0 to i32
  %5 = bitcast i32 0 to i32
  %6 = bitcast i32 0 to i32
  %7 = bitcast i32 0 to i32
  %8 = bitcast i32 0 to i32
  %9 = bitcast i32 0 to i32
  %10 = bitcast i32 0 to i32
  %11 = bitcast i32 0 to i32
  %12 = bitcast i32 0 to i32
  %13 = bitcast i32 0 to i32
  %14 = bitcast i32 0 to i32
  %15 = bitcast i32 0 to i32
  %16 = bitcast i32 0 to i32
  %17 = bitcast i32 0 to i32
  %18 = bitcast i32 0 to i32
  %19 = bitcast i32 0 to i32
  %20 = bitcast i32 0 to i32
  %21 = bitcast i32 0 to i32
  %22 = bitcast i32 0 to i32
  %23 = bitcast i32 0 to i32
  %24 = bitcast i32 0 to i32
  %25 = bitcast i32 0 to i32
  %26 = bitcast i32 0 to i32
  %27 = bitcast i32 0 to i32
  %28 = bitcast i32 0 to i32
  %29 = bitcast i32 0 to i32
  %30 = bitcast i32 0 to i32
  %31 = bitcast i32 0 to i32
  %32 = bitcast i32 0 to i32
  %33 = bitcast i32 0 to i32
  %34 = bitcast i32 0 to i32
  %35 = bitcast i32 0 to i32
  %36 = bitcast i32 0 to i32
  %37 = bitcast i32 0 to i32
  %38 = bitcast i32 0 to i32
  %39 = bitcast i32 0 to i32
  %40 = bitcast i32 0 to i32
  %41 = bitcast i32 0 to i32
  %42 = bitcast i32 0 to i32
  %43 = bitcast i32 0 to i32
  %44 = bitcast i32 0 to i32
  %45 = bitcast i32 0 to i32
  %46 = bitcast i32 0 to i32
  %47 = bitcast i32 0 to i32
  %48 = bitcast i32 0 to i32
  %49 = bitcast i32 0 to i32
  %50 = bitcast i32 0 to i32
  %51 = bitcast i32 0 to i32
  %52 = bitcast i32 0 to i32
  %53 = bitcast i32 0 to i32
  %54 = bitcast i32 0 to i32
  %55 = bitcast i32 0 to i32
  %56 = bitcast i32 0 to i32
  %57 = bitcast i32 0 to i32
  %58 = bitcast i32 0 to i32
  %59 = bitcast i32 0 to i32
  %60 = bitcast i32 0 to i32
  %61 = bitcast i32 0 to i32
  %62 = bitcast i32 0 to i32
  %63 = bitcast i32 0 to i32
  %64 = bitcast i32 0 to i32
  %65 = bitcast i32 0 to i32
  %66 = bitcast i32 0 to i32
  %67 = bitcast i32 0 to i32
  %68 = bitcast i32 0 to i32
  %69 = bitcast i32 0 to i32
  %70 = bitcast i32 0 to i32
  %71 = bitcast i32 0 to i32
  %72 = bitcast i32 0 to i32
  %73 = bitcast i32 0 to i32
  %74 = bitcast i32 0 to i32
  %75 = bitcast i32 0 to i32
  %76 = bitcast i32 0 to i32
  %77 = bitcast i32 0 to i32
  %78 = bitcast i32 0 to i32
  %79 = bitcast i32 0 to i32
  %80 = bitcast i32 0 to i32
  %81 = bitcast i32 0 to i32
  %82 = bitcast i32 0 to i32
  %83 = bitcast i32 0 to i32
  %84 = bitcast i32 0 to i32
  %85 = bitcast i32 0 to i32
  %86 = bitcast i32 0 to i32
  %87 = bitcast i32 0 to i32
  %88 = bitcast i32 0 to i32
  %89 = bitcast i32 0 to i32
  %90 = bitcast i32 0 to i32
  %91 = bitcast i32 0 to i32
  %92 = bitcast i32 0 to i32
  %93 = bitcast i32 0 to i32
  %94 = bitcast i32 0 to i32
  %95 = bitcast i32 0 to i32
  %96 = bitcast i32 0 to i32
  %97 = bitcast i32 0 to i32
  %98 = bitcast i32 0 to i32

  ; CHECK:  store i32 -1, i32* @x, align 4
  store i32 -1, i32* @x, align 4
  ret i32 0
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!11, !13}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.4", isOptimized: true, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, subprograms: !3, globals: !2, imports: !2)
!1 = !DIFile(filename: "test.c", directory: "/home/tmp")
!2 = !{}
!3 = !{!4}
!4 = distinct !DISubprogram(name: "test_within_limit", line: 3, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 4, file: !1, scope: !5, type: !6, variables: !2)
!5 = !DIFile(filename: "test.c", directory: "/home/tmp")
!6 = !DISubroutineType(types: !7)
!7 = !{!8}
!8 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !{!10}
!10 = !DILocalVariable(name: "x", scope: !4, type: !8)
!11 = !{i32 2, !"Dwarf Version", i32 4}
!12 = !{i32* undef}

!13 = !{i32 1, !"Debug Info Version", i32 3}
