; RUN: rm -rf %t && mkdir -p %t
; RUN: echo '!16 = !{!"%/t/global-ctor.ll", !0}' > %t/1
; RUN: cat %s %t/1 > %t/2
; RUN: opt -insert-gcov-profiling -disable-output < %t/2
; RUN: not grep '_GLOBAL__sub_I_global-ctor' %t/global-ctor.gcno
; RUN: rm %t/global-ctor.gcno

; RUN: opt -passes=insert-gcov-profiling -disable-output < %t/2
; RUN: not grep '_GLOBAL__sub_I_global-ctor' %t/global-ctor.gcno
; RUN: rm %t/global-ctor.gcno

@x = global i32 0, align 4
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_global-ctor.ll, i8* null }]

; Function Attrs: nounwind
define internal void @__cxx_global_var_init() #0 section ".text.startup" !dbg !4 {
entry:
  br label %0

; <label>:0                                       ; preds = %entry
  %call = call i32 @_Z1fv(), !dbg !13
  store i32 %call, i32* @x, align 4, !dbg !13
  ret void, !dbg !13
}

declare i32 @_Z1fv() #1

; Function Attrs: nounwind
define internal void @_GLOBAL__sub_I_global-ctor.ll() #0 section ".text.startup" {
entry:
  br label %0

; <label>:0                                       ; preds = %entry
  call void @__cxx_global_var_init(), !dbg !14
  ret void, !dbg !14
}

attributes #0 = { nounwind }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10, !11}
!llvm.gcov = !{!16}
!llvm.ident = !{!12}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.5.0 (trunk 210217)", isOptimized: false, emissionKind: LineTablesOnly, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "<stdin>", directory: "/home/nlewycky")
!2 = !{}
!4 = distinct !DISubprogram(name: "__cxx_global_var_init", line: 2, isLocal: true, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 2, file: !5, scope: !6, type: !7, retainedNodes: !2)
!5 = !DIFile(filename: "global-ctor.ll", directory: "/home/nlewycky")
!6 = !DIFile(filename: "global-ctor.ll", directory: "/home/nlewycky")
!7 = !DISubroutineType(types: !2)
!8 = distinct !DISubprogram(name: "", linkageName: "_GLOBAL__sub_I_global-ctor.ll", isLocal: true, isDefinition: true, virtualIndex: 6, flags: DIFlagArtificial, isOptimized: false, unit: !0, file: !1, scope: !9, type: !7, retainedNodes: !2)
!9 = !DIFile(filename: "<stdin>", directory: "/home/nlewycky")
!10 = !{i32 2, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{!"clang version 3.5.0 (trunk 210217)"}
!13 = !DILocation(line: 2, scope: !4)
!14 = !DILocation(line: 0, scope: !15)
!15 = !DILexicalBlockFile(discriminator: 0, file: !5, scope: !8)
