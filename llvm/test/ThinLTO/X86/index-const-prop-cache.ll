; Check that we correctly handle 'ReadOnly' attribute when computing cache key
; RUN: opt -module-summary -module-hash %s -o %t1.bc
; RUN: opt -module-summary -module-hash %p/Inputs/index-const-prop-cache-foo.ll -o %t2.bc
; RUN: opt -module-summary -module-hash %p/Inputs/index-const-prop-cache-test1.ll -o %t3.bc
; RUN: opt -module-summary -module-hash %p/Inputs/index-const-prop-cache-test2.ll -o %t4.bc
; RUN: rm -Rf %t.cache && mkdir %t.cache

; Here @gFoo variable is writeable
; RUN: llvm-lto -thinlto-action=run %t1.bc %t4.bc %t2.bc \
; RUN:    -exported-symbol=main -exported-symbol=test -thinlto-cache-dir=%t.cache
; RUN: ls %t.cache/llvmcache-* | count 3

; Now gFoo is read-only and all modules should get different cache keys.
; RUN: llvm-lto -thinlto-action=run %t1.bc %t3.bc %t2.bc \
; RUN:    -exported-symbol=main -exported-symbol=test -thinlto-cache-dir=%t.cache
; RUN: ls %t.cache/llvmcache-* | count 6

; Do the same but with llvm-lto2
; RUN: rm -Rf %t.cache && mkdir %t.cache
; RUN: llvm-lto2 run  %t1.bc %t4.bc %t2.bc -cache-dir=%t.cache -o %t5 \
; RUN:  -r=%t1.bc,main,plx -r=%t1.bc,foo,l \
; RUN:  -r=%t4.bc,test,plx -r=%t4.bc,foo,l -r=%t4.bc,bar,l \
; RUN:  -r=%t2.bc,foo,pl -r=%t2.bc,bar,pl -r=%t2.bc,rand,
; RUN: ls %t.cache/llvmcache-* | count 3

; RUN: llvm-lto2 run %t1.bc %t3.bc %t2.bc -cache-dir %t.cache -o %t6 \
; RUN:  -r=%t1.bc,main,plx -r=%t1.bc,foo,l \
; RUN:  -r=%t3.bc,test,plx -r=%t3.bc,foo,l \
; RUN:  -r=%t2.bc,foo,pl -r=%t2.bc,bar,pl -r=%t2.bc,rand,
; RUN: ls %t.cache/llvmcache-* | count 6

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind ssp uwtable
define i32 @main() local_unnamed_addr {
  %1 = tail call i32 (...) @foo()
  ret i32 %1
}

declare i32 @foo(...) local_unnamed_addr
