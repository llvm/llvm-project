; UNSUPPORTED: system-windows
;; Unsupported on Windows due to difficulty with escaping "opt" across platforms.
;; lit substitutes 'opt' with /path/to/opt.

; RUN: rm -rf %t && mkdir %t && cd %t

;; Copy IR from import-constant.ll since it generates all the temps
; RUN: opt -thinlto-bc %s -o 1.bc
; RUN: opt -thinlto-bc %p/Inputs/import-constant.ll -o 2.bc

;; Create the .all dir with save-temps saving everything, this will be used to compare
;; with the output from individualized save-temps later
; RUN: mkdir all all2 build subset subset2
; RUN: llvm-lto2 run 1.bc 2.bc -o all/a.out \
; RUN:    -import-constants-with-refs -r=1.bc,main,plx -r=1.bc,_Z6getObjv,l \
; RUN:    -r=2.bc,_Z6getObjv,pl -r=2.bc,val,pl -r=2.bc,outer,pl \
; RUN:    -save-temps

;; The next 8 blocks follow this structure:
;; for each option of save-temps=
;;   Run lto and generate files
;;   Make sure a.out exists and is correct (by diff-ing)
;;     this is the only file that should recur between runs
;;   (Also, for some stages, copy the generated files to subset2 to check composability later)
;;   Move files that were expected to be generated to all2
;;   Make sure there's no unexpected extra files
;; After that, we'll diff all and all2 to make sure all contents are identical

;; Check preopt
; RUN: llvm-lto2 run 1.bc 2.bc -o build/a.out \
; RUN:    -import-constants-with-refs -r=1.bc,main,plx -r=1.bc,_Z6getObjv,l \
; RUN:    -r=2.bc,_Z6getObjv,pl -r=2.bc,val,pl -r=2.bc,outer,pl \
; RUN:    -select-save-temps=preopt
; RUN: cmp all/a.out.1 build/a.out.1 && rm -f build/a.out.1
; RUN: cmp all/a.out.2 build/a.out.2 && rm -f build/a.out.2
; RUN: cp build/*.0.preopt.* subset2
; RUN: mv build/*.0.preopt.* all2
; RUN: ls build | count 0

;; Check promote
; RUN: rm -f all2/*.1.promote*
; RUN: llvm-lto2 run 1.bc 2.bc -o build/a.out \
; RUN:    -import-constants-with-refs -r=1.bc,main,plx -r=1.bc,_Z6getObjv,l \
; RUN:    -r=2.bc,_Z6getObjv,pl -r=2.bc,val,pl -r=2.bc,outer,pl \
; RUN:    -select-save-temps=promote
; RUN: cmp all/a.out.1 build/a.out.1 && rm -f build/a.out.1
; RUN: cmp all/a.out.2 build/a.out.2 && rm -f build/a.out.2
; RUN: mv build/*.1.promote* all2
; RUN: ls build | count 0

;; Check internalize
; RUN: llvm-lto2 run 1.bc 2.bc -o build/a.out \
; RUN:    -import-constants-with-refs -r=1.bc,main,plx -r=1.bc,_Z6getObjv,l \
; RUN:    -r=2.bc,_Z6getObjv,pl -r=2.bc,val,pl -r=2.bc,outer,pl \
; RUN:    -select-save-temps=internalize
; RUN: cmp all/a.out.1 build/a.out.1 && rm -f build/a.out.1
; RUN: cmp all/a.out.2 build/a.out.2 && rm -f build/a.out.2
; RUN: mv build/*.2.internalize* all2
; RUN: ls build | count 0

;; Check import
; RUN: llvm-lto2 run 1.bc 2.bc -o build/a.out \
; RUN:    -import-constants-with-refs -r=1.bc,main,plx -r=1.bc,_Z6getObjv,l \
; RUN:    -r=2.bc,_Z6getObjv,pl -r=2.bc,val,pl -r=2.bc,outer,pl \
; RUN:    -select-save-temps=import
; RUN: cmp all/a.out.1 build/a.out.1 && rm -f build/a.out.1
; RUN: cmp all/a.out.2 build/a.out.2 && rm -f build/a.out.2
; RUN: mv build/*.3.import* all2
; RUN: ls build | count 0

;; Check opt
; RUN: llvm-lto2 run 1.bc 2.bc -o build/a.out \
; RUN:    -import-constants-with-refs -r=1.bc,main,plx -r=1.bc,_Z6getObjv,l \
; RUN:    -r=2.bc,_Z6getObjv,pl -r=2.bc,val,pl -r=2.bc,outer,pl \
; RUN:    -select-save-temps=\opt
; RUN: cmp all/a.out.1 build/a.out.1 && rm -f build/a.out.1
; RUN: cmp all/a.out.2 build/a.out.2 && rm -f build/a.out.2
; RUN: cp build/*.4.opt* subset2
; RUN: mv build/*.4.opt* all2
; RUN: ls build | count 0

;; Check precodegen
; RUN: llvm-lto2 run 1.bc 2.bc -o build/a.out \
; RUN:    -import-constants-with-refs -r=1.bc,main,plx -r=1.bc,_Z6getObjv,l \
; RUN:    -r=2.bc,_Z6getObjv,pl -r=2.bc,val,pl -r=2.bc,outer,pl \
; RUN:    -select-save-temps=precodegen
; RUN: cmp all/a.out.1 build/a.out.1 && rm -f build/a.out.1
; RUN: cmp all/a.out.2 build/a.out.2 && rm -f build/a.out.2
; RUN: mv build/*.5.precodegen* all2
; RUN: ls build | count 0

;; Check combinedindex
; RUN: llvm-lto2 run 1.bc 2.bc -o build/a.out \
; RUN:    -import-constants-with-refs -r=1.bc,main,plx -r=1.bc,_Z6getObjv,l \
; RUN:    -r=2.bc,_Z6getObjv,pl -r=2.bc,val,pl -r=2.bc,outer,pl \
; RUN:    -select-save-temps=combinedindex
; RUN: cmp all/a.out.1 build/a.out.1 && rm -f build/a.out.1
; RUN: cmp all/a.out.2 build/a.out.2 && rm -f build/a.out.2
; RUN: cp build/*.index.bc subset2
; RUN: cp build/*.index.dot subset2
; RUN: mv build/*.index.bc all2
; RUN: mv build/*.index.dot all2
; RUN: ls build | count 0

;; Check resolution
; RUN: llvm-lto2 run 1.bc 2.bc -o build/a.out \
; RUN:    -import-constants-with-refs -r=1.bc,main,plx -r=1.bc,_Z6getObjv,l \
; RUN:    -r=2.bc,_Z6getObjv,pl -r=2.bc,val,pl -r=2.bc,outer,pl \
; RUN:    -select-save-temps=resolution
;; all2 needs at least 1 copy of a.out, move it over now since its the last block
; RUN: mv build/a.out.1 build/a.out.2 all2
; RUN: mv build/*.resolution.txt all2
; RUN: ls build | count 0

;; If no files were left out from individual stages, the .all2 dir should be identical to .all
; RUN: diff -r all all2

;; Check multi-stage composability
;; Similar to the above, but do it with a subset instead.
;; .all -> .subset, .all2 -> .subset2
; RUN: llvm-lto2 run 1.bc 2.bc -o subset/a.out \
; RUN:    -import-constants-with-refs -r=1.bc,main,plx -r=1.bc,_Z6getObjv,l \
; RUN:    -r=2.bc,_Z6getObjv,pl -r=2.bc,val,pl -r=2.bc,outer,pl \
; RUN:    -select-save-temps=preopt,combinedindex,\opt
; RUN: cmp all/a.out.1 subset/a.out.1 && rm -f subset/a.out.1
; RUN: cmp all/a.out.2 subset/a.out.2 && rm -f subset/a.out.2
; RUN: diff -r subset subset2

;; Check error messages
; RUN: not llvm-lto2 run 1.bc 2.bc -o build/a.out \
; RUN:    -import-constants-with-refs -r=1.bc,main,plx -r=1.bc,_Z6getObjv,l \
; RUN:    -r=2.bc,_Z6getObjv,pl -r=2.bc,val,pl -r=2.bc,outer,pl \
; RUN:    -select-save-temps=prelink 2>&1 \
; RUN: | FileCheck %s --check-prefix=ERR1
; ERR1: invalid -select-save-temps argument: prelink

; RUN: not llvm-lto2 run 1.bc 2.bc -o build/a.out \
; RUN:    -import-constants-with-refs -r=1.bc,main,plx -r=1.bc,_Z6getObjv,l \
; RUN:    -r=2.bc,_Z6getObjv,pl -r=2.bc,val,pl -r=2.bc,outer,pl \
; RUN:    -select-save-temps=preopt -save-temps 2>&1 \
; RUN: | FileCheck %s --check-prefix=ERR2
; ERR2: -save-temps cannot be specified with -select-save-temps

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.S = type { i32, i32, i32* }

define dso_local i32 @main() local_unnamed_addr {
entry:
  %call = tail call %struct.S* @_Z6getObjv()
  %d = getelementptr inbounds %struct.S, %struct.S* %call, i64 0, i32 0
  %0 = load i32, i32* %d, align 8
  %v = getelementptr inbounds %struct.S, %struct.S* %call, i64 0, i32 1
  %1 = load i32, i32* %v, align 4
  %add = add nsw i32 %1, %0
  ret i32 %add
}

declare dso_local %struct.S* @_Z6getObjv() local_unnamed_addr
