; This test verifies whether we can outline a singleton instance (i.e., an instance that does not repeat)
; by running two codegen rounds.
; This test also verifies if caches for the two-round codegens are correctly working.

; REQUIRES: asserts
; RUN: rm -rf %t
; RUN: split-file %s %t

; 0. Base case without a cache.
; Verify each outlining instance is singleton with the global outlining for thinlto.
; They will be identical, which can be folded by the linker with ICF.
; RUN: opt -module-hash -module-summary %t/thin-one.ll -o %t/thin-one.bc
; RUN: opt -module-hash -module-summary %t/thin-two.ll -o %t/thin-two.bc
; RUN: llvm-lto2 run %t/thin-one.bc %t/thin-two.bc -o %t/thinlto \
; RUN:  -r %t/thin-one.bc,_f3,px -r %t/thin-one.bc,_g,x \
; RUN:  -r %t/thin-two.bc,_f1,px -r %t/thin-two.bc,_f2,px -r %t/thin-two.bc,_g,x \
; RUN:  -codegen-data-thinlto-two-rounds

; thin-one.ll will have one outlining instance (matched in the global outlined hash tree)
; RUN: llvm-objdump -d %t/thinlto.1 | FileCheck %s --check-prefix=THINLTO-1
; THINLTO-1: _OUTLINED_FUNCTION{{.*}}>:
; THINLTO-1-NEXT:  mov
; THINLTO-1-NEXT:  mov
; THINLTO-1-NEXT:  b

; thin-two.ll will have two outlining instances (matched in the global outlined hash tree)
; RUN: llvm-objdump -d %t/thinlto.2 | FileCheck %s --check-prefix=THINLTO-2
; THINLTO-2: _OUTLINED_FUNCTION{{.*}}>:
; THINLTO-2-NEXT:  mov
; THINLTO-2-NEXT:  mov
; THINLTO-2-NEXT:  b
; THINLTO-2: _OUTLINED_FUNCTION{{.*}}>:
; THINLTO-2-NEXT:  mov
; THINLTO-2-NEXT:  mov
; THINLTO-2-NEXT:  b

; 1. Run this with a cache for the first time.
; RUN: rm -rf %t.cache
; RUN: llvm-lto2 run %t/thin-one.bc %t/thin-two.bc -o %t/thinlto-cold \
; RUN:  -r %t/thin-one.bc,_f3,px -r %t/thin-one.bc,_g,x \
; RUN:  -r %t/thin-two.bc,_f1,px -r %t/thin-two.bc,_f2,px -r %t/thin-two.bc,_g,x \
; RUN:  -codegen-data-thinlto-two-rounds -cache-dir %t.cache -debug-only=lto -thinlto-threads 1 > %t.log-cold.txt 2>&1
; RUN: cat %t.log-cold.txt | FileCheck %s --check-prefix=COLD
; diff %t/thinlto.1 %t/thinlto-cold.1
; diff %t/thinlto.2 %t/thinlto-cold.2

; COLD: [FirstRound] Cache Miss for {{.*}}thin-one.bc
; COLD: [FirstRound] Cache Miss for {{.*}}thin-two.bc
; COLD: [SecondRound] Cache Miss for {{.*}}thin-one.bc
; COLD: [SecondRound] Cache Miss for {{.*}}thin-two.bc

; There are two input bitcode files and each one is operated with 3 caches:
; CG/IR caches for the first round and the second round CG cache.
; So the total number of files are 2 * 3 = 6.
; RUN: ls %t.cache | count 6

; 2. Without any changes, simply re-running it will hit the cache.
; RUN: llvm-lto2 run %t/thin-one.bc %t/thin-two.bc -o %t/thinlto-warm \
; RUN:  -r %t/thin-one.bc,_f3,px -r %t/thin-one.bc,_g,x \
; RUN:  -r %t/thin-two.bc,_f1,px -r %t/thin-two.bc,_f2,px -r %t/thin-two.bc,_g,x \
; RUN:  -codegen-data-thinlto-two-rounds -cache-dir %t.cache -debug-only=lto -thinlto-threads 1 > %t.log-warm.txt 2>&1
; RUN: cat %t.log-warm.txt | FileCheck %s --check-prefix=WARM
; diff %t/thinlto.1 %t/thinlto-warm.1
; diff %t/thinlto.2 %t/thinlto-warm.2

; WARM-NOT: Cache Miss

; 3. Assume thin-one.ll has been modified to thin-one-modified.ll.
; The merged CG data remains unchanged as this modification does not affect the hash tree built from thin-two.bc.
; Therefore, both the first and second round runs update only this module.
; RUN: opt -module-hash -module-summary %t/thin-one-modified.ll -o %t/thin-one.bc
; RUN: llvm-lto2 run %t/thin-one.bc %t/thin-two.bc -o %t/thinlto-warm-modified \
; RUN:  -r %t/thin-one.bc,_f3,px -r %t/thin-one.bc,_g,x \
; RUN:  -r %t/thin-two.bc,_f1,px -r %t/thin-two.bc,_f2,px -r %t/thin-two.bc,_g,x \
; RUN:  -codegen-data-thinlto-two-rounds -cache-dir %t.cache -debug-only=lto -thinlto-threads 1 > %t.log-warm-modified.txt 2>&1
; RUN: cat %t.log-warm-modified.txt | FileCheck %s --check-prefix=WARM-MODIFIED
; diff %t/thinlto.1 %t/thinlto-warm-modified.1
; diff %t/thinlto.2 %t/thinlto-warm-modified.2

; WARM-MODIFIED: [FirstRound] Cache Miss for {{.*}}thin-one.bc
; WARM-MODIFIED-NOT: [FirstRound] Cache Miss for {{.*}}thin-two.bc
; WARM-MODIFIED: [SecondRound] Cache Miss for {{.*}}thin-one.bc
; WARM-MODIFIED-NOT: [SecondRound] Cache Miss for {{.*}}thin-two.bc

; 4. Additionally, thin-two.ll has been modified to thin-two-modified.ll.
; In this case, the merged CG data, which is global, is updated.
; Although the first round run updates only the thin-two.bc module,
; as the module thin-one.bc remains the same as in step 3 above,
; the second round run will update all modules, resulting in different binaries.
; RUN: opt -module-hash -module-summary %t/thin-one-modified.ll -o %t/thin-one.bc
; RUN: opt -module-hash -module-summary %t/thin-two-modified.ll -o %t/thin-two.bc
; RUN: llvm-lto2 run %t/thin-one.bc %t/thin-two.bc -o %t/thinlto-warm-modified-all \
; RUN:  -r %t/thin-one.bc,_f3,px -r %t/thin-one.bc,_g,x \
; RUN:  -r %t/thin-two.bc,_f1,px -r %t/thin-two.bc,_f2,px -r %t/thin-two.bc,_g,x \
; RUN:  -codegen-data-thinlto-two-rounds -cache-dir %t.cache -debug-only=lto -thinlto-threads 1 > %t.log-warm-modified-all.txt 2>&1
; RUN: cat %t.log-warm-modified-all.txt | FileCheck %s --check-prefix=WARM-MODIFIED-ALL
; RUN: not diff %t/thinlto.1 %t/thinlto-warm-modified-all.1
; RUN: not diff %t/thinlto.2 %t/thinlto-warm-modified-all.2

; WARM-MODIFIED-ALL-NOT: [FirstRound] Cache Miss for {{.*}}thin-one.bc
; WARM-MODIFIED-ALL: [FirstRound] Cache Miss for {{.*}}thin-two.bc
; WARM-MODIFIED-ALL: [SecondRound] Cache Miss for {{.*}}thin-one.bc
; WARM-MODIFIED-ALL: [SecondRound] Cache Miss for {{.*}}thin-two.bc

; thin-one-modified.ll won't be outlined.
; RUN: llvm-objdump -d %t/thinlto-warm-modified-all.1 | FileCheck %s --check-prefix=THINLTO-1-MODIFIED-ALL
; THINLTO-1-MODIFIED-ALL-NOT: _OUTLINED_FUNCTION{{.*}}>:

; thin-two-modified.ll will have two (longer) outlining instances (matched in the global outlined hash tree)
; RUN: llvm-objdump -d %t/thinlto-warm-modified-all.2| FileCheck %s --check-prefix=THINLTO-2-MODIFIED-ALL
; THINLTO-2-MODIFIED-ALL: _OUTLINED_FUNCTION{{.*}}>:
; THINLTO-2-MODIFIED-ALL:  mov
; THINLTO-2-MODIFIED-ALL:  mov
; THINLTO-2-MODIFIED-ALL:  mov
; THINLTO-2-MODIFIED-ALL:  b
; THINLTO-2-MODIFIED-ALL: _OUTLINED_FUNCTION{{.*}}>:
; THINLTO-2-MODIFIED-ALL:  mov
; THINLTO-2-MODIFIED-ALL:  mov
; THINLTO-2-MODIFIED-ALL:  mov
; THINLTO-2-MODIFIED-ALL:  b

; 5. Re-running it will hit the cache.
; RUN: llvm-lto2 run %t/thin-one.bc %t/thin-two.bc -o %t/thinlto-warm-again \
; RUN:  -r %t/thin-one.bc,_f3,px -r %t/thin-one.bc,_g,x \
; RUN:  -r %t/thin-two.bc,_f1,px -r %t/thin-two.bc,_f2,px -r %t/thin-two.bc,_g,x \
; RUN:  -codegen-data-thinlto-two-rounds -cache-dir %t.cache -debug-only=lto -thinlto-threads 1 > %t.log-warm-again.txt 2>&1
; RUN: cat %t.log-warm-again.txt | FileCheck %s --check-prefix=WARM-AGAIN
; RUN: diff %t/thinlto-warm-modified-all.1 %t/thinlto-warm-again.1
; RUN: diff %t/thinlto-warm-modified-all.2 %t/thinlto-warm-again.2

; WARM-AGAIN-NOT: Cache Miss

;--- thin-one.ll
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-darwin"

declare i32 @g(i32, i32, i32)
define i32 @f3() minsize {
  %1 = call i32 @g(i32 30, i32 1, i32 2);
 ret i32 %1
}

;--- thin-one-modified.ll
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-darwin"

declare i32 @g(i32, i32, i32)
define i32 @f3() minsize {
  %1 = call i32 @g(i32 31, i32 1, i32 2);
 ret i32 %1
}

;--- thin-two.ll
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-darwin"

declare i32 @g(i32, i32, i32)
define i32 @f1() minsize {
  %1 = call i32 @g(i32 10, i32 1, i32 2);
  ret i32 %1
}
define i32 @f2() minsize {
  %1 = call i32 @g(i32 20, i32 1, i32 2);
  ret i32 %1
}

;--- thin-two-modified.ll
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-darwin"

declare i32 @g(i32, i32, i32)
define i32 @f1() minsize {
  %1 = call i32 @g(i32 10, i32 1, i32 2);
  ret i32 %1
}
define i32 @f2() minsize {
  %1 = call i32 @g(i32 10, i32 1, i32 2);
  ret i32 %1
}
