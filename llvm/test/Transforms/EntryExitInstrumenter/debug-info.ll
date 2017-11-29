; RUN: opt -passes="function(ee-instrument),cgscc(inline),function(post-inline-ee-instrument)" -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @f(i32 %x) #0 !dbg !7 {
entry:
  %x.addr = alloca i32, align 4
  store i32 %x, i32* %x.addr, align 4
  ret i32 42, !dbg !12

; CHECK-LABEL: define i32 @f(i32 %x)
; CHECK: call i8* @llvm.returnaddress(i32 0), !dbg ![[ENTRYLOC:[0-9]+]]
; CHECK: call void @__cyg_profile_func_enter{{.*}}, !dbg ![[ENTRYLOC]]

; CHECK: call i8* @llvm.returnaddress(i32 0), !dbg ![[EXITLOC:[0-9]+]]
; CHECK: call void @__cyg_profile_func_exit{{.*}}, !dbg ![[EXITLOC]]
; CHECK: ret i32 42, !dbg ![[EXITLOC]]
}

; CHECK: ![[SP:[0-9]+]] = distinct !DISubprogram(name: "f"
; CHECK: ![[ENTRYLOC]] = !DILocation(line: 2, scope: ![[SP]])
; CHECK: ![[EXITLOC]] = !DILocation(line: 4, column: 3, scope: ![[SP]])

attributes #0 = { "instrument-function-entry"="__cyg_profile_func_enter" "instrument-function-exit"="__cyg_profile_func_exit" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 6.0.0 (trunk 319007) (llvm/trunk 319050)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "a.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 6.0.0 (trunk 319007) (llvm/trunk 319050)"}
!7 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 2, type: !8, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DILocalVariable(name: "x", arg: 1, scope: !7, file: !1, line: 2, type: !10)
!12 = !DILocation(line: 4, column: 3, scope: !7)
