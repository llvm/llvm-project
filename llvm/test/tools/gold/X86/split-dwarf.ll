; RUN: rm -rf %t && mkdir -p %t
; RUN: opt -module-summary %s -o %t/split-dwarf.o
; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext  \
; RUN:    -m elf_x86_64 \
; RUN:    --plugin-opt=thinlto \
; RUN:    --plugin-opt=objcopy=%llvm-objcopy \
; RUN:    --plugin-opt=dwo_dir=%t/dwo_dir \
; RUN:    %t/split-dwarf.o --shared -o %t/split-dwarf

; RUN: llvm-dwarfdump -debug-info %t/split-dwarf | FileCheck %s
; CHECK: DW_AT_GNU_dwo_name{{.*}}dwo_dir/split-dwarf.{{.*}}
; CHECK-NOT: DW_TAG_subprogram
; RUN: llvm-dwarfdump -debug-info %t/dwo_dir/split-dwarf.* | FileCheck --check-prefix DWOCHECK %s
; DWOCHECK: DW_AT_GNU_dwo_name{{.*}}dwo_dir/split-dwarf.o{{.*}}
; DWOCHECK: DW_AT_name{{.*}}split-dwarf.c
; DWOCHECK: DW_TAG_subprogram

; RUN:rm -rf %t/dwo_dir
; RUN: opt  %s -o %t/split-dwarf.o
; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext  \
; RUN:    -m elf_x86_64 \
; RUN:    --plugin-opt=thinlto \
; RUN:    --plugin-opt=objcopy=%llvm-objcopy \
; RUN:    --plugin-opt=dwo_dir=%t/dwo_dir \
; RUN:    %t/split-dwarf.o --shared -o %t/split-dwarf

; RUN: llvm-dwarfdump -debug-info %t/split-dwarf | FileCheck --check-prefix LTOCHECK %s
; LTOCHECK: DW_AT_GNU_dwo_name{{.*}}dwo_dir/ld-temp.{{.*}}
; LTOCHECK-NOT: DW_TAG_subprogram
; RUN: llvm-dwarfdump -debug-info %t/dwo_dir/ld-temp.* | FileCheck --check-prefix LTODWOCHECK %s
; LTODWOCHECK: DW_AT_GNU_dwo_name{{.*}}dwo_dir/ld-temp.o{{.*}}
; LTODWOCHECK: DW_AT_name{{.*}}split-dwarf.c
; LTODWOCHECK: DW_TAG_subprogram

; ModuleID = 'split-dwarf.c'
source_filename = "split-dwarf.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @split_dwarf() #0 !dbg !7 {
entry:
  ret i32 0, !dbg !11
}

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 7.0.0 (https://github.com/llvm-mirror/clang.git b641d31365414ba3ea0305fdaa80369a9efb6bd9) (https://github.com/llvm-mirror/llvm.git 6165a776d1a8bb181be93f2dc97088f7a1abc405)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "split-dwarf.c", directory: "/usr/local/google/home/yunlian/dwp/build/bin")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 7.0.0 (https://github.com/llvm-mirror/clang.git b641d31365414ba3ea0305fdaa80369a9efb6bd9) (https://github.com/llvm-mirror/llvm.git 6165a776d1a8bb181be93f2dc97088f7a1abc405)"}
!7 = distinct !DISubprogram(name: "split_dwarf", scope: !1, file: !1, line: 1, type: !8, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: false, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{!10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DILocation(line: 2, column: 2, scope: !7)
