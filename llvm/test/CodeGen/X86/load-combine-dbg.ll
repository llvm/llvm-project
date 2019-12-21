; RUN: llc -O0 < %s -mtriple=x86_64-unknown | FileCheck %s
; This was extracted from a swift debugger stepping testcase and checks that the
;   fold (zext (load x)) -> (zext (truncate (zextload x)))
; rule propagates the SDLoc of the load to the zextload.

; CHECK: .loc {{.*}} main.swift:100
; CHECK-NOT: .loc
; CHECK: .loc {{.*}} main.swift:200
; CHECK-NOT: .loc
; CHECK: .loc {{.*}} main.swift:300
; CHECK-NOT: .loc
declare void @foo(double)

define i32 @zext_load(i32* %arg) !dbg !30 {
  %1 = bitcast i32* %arg to i8*
  %2 = getelementptr inbounds i8, i8* %1, i32 1
  %3 = load i8, i8* %2, align 1, !dbg !100
  %4 = uitofp i8 %3 to double, !dbg !200
  call void @foo(double %4), !dbg !200
  %5 = zext i8 %3 to i32, !dbg !300
  ret i32 %5
}
!llvm.dbg.cu = !{!1}
!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_Swift, file: !3, isOptimized: false, emissionKind: FullDebug)
!2 = !DIModule(scope: null, name: "test", includePath: "", sysroot: "/")
!3 = !DIFile(filename: "main.swift", directory: "/")

!30 = distinct !DISubprogram(name: "main", scope: !2, file: !3, line: 1, type: !31, isLocal: false, isDefinition: true, isOptimized: false, unit: !1)
!31 = !DISubroutineType(types: !32)
!32 = !{}

!100 = !DILocation(line: 100, scope: !30)
!200 = !DILocation(line: 200, scope: !30)
!300 = !DILocation(line: 300, scope: !30)
