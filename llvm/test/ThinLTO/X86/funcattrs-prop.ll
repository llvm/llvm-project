; RUN: split-file %s %t
; RUN: opt -module-summary %t/a.ll -o %t/a.bc
; RUN: opt -module-summary %t/b.ll -o %t/b.bc
; RUN: opt -module-summary %t/c.ll -o %t/c.bc

;; ThinLTO Function attribute propagation uses the prevailing symbol to propagate attributes to its callers. 
;; Interposable (linkonce and weak) linkages are fair game given we know the prevailing copy will be used in the final binary.
; RUN: llvm-lto2 run -disable-thinlto-funcattrs=0 %t/a.bc %t/b.bc %t/c.bc -o %t1 -save-temps \
; RUN:   -r=%t/a.bc,call_extern,plx -r=%t/a.bc,call_linkonceodr,plx -r=%t/a.bc,call_weakodr,plx -r=%t/a.bc,call_linkonce,plx -r=%t/a.bc,call_weak,plx -r=%t/a.bc,call_linkonce_may_unwind,plx -r=%t/a.bc,call_weak_may_unwind,plx \
; RUN:   -r=%t/a.bc,extern, -r=%t/a.bc,linkonceodr, -r=%t/a.bc,weakodr, -r=%t/a.bc,linkonce, -r=%t/a.bc,weak, -r=%t/a.bc,linkonce_may_unwind, -r=%t/a.bc,weak_may_unwind, \
; RUN:   -r=%t/b.bc,extern,p -r=%t/b.bc,linkonceodr,p -r=%t/b.bc,weakodr,p -r=%t/b.bc,linkonce,p -r=%t/b.bc,weak,p -r=%t/b.bc,linkonce_may_unwind,p -r=%t/b.bc,weak_may_unwind, \
; RUN:   -r=%t/c.bc,extern, -r=%t/c.bc,linkonceodr, -r=%t/c.bc,weakodr, -r=%t/c.bc,linkonce, -r=%t/c.bc,weak, -r=%t/c.bc,linkonce_may_unwind, -r=%t/c.bc,weak_may_unwind,p -r=%t/c.bc,may_throw,

; RUN: llvm-dis %t1.1.3.import.bc -o - | FileCheck %s

;--- a.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

;; These functions are identical between b.ll and c.ll
declare void @extern()
declare void @linkonceodr()
declare void @weakodr()

declare void @linkonce()
declare void @weak()

;; b.ll contains non-throwing copies of these functions
;; c.ll contains throwing copies of these functions
declare void @linkonce_may_unwind()
declare void @weak_may_unwind()

; CHECK: define dso_local void @call_extern() [[ATTR_NOUNWIND:#[0-9]+]]
define void @call_extern() {
    call void @extern()
    ret void
}

; CHECK: define dso_local void @call_linkonceodr() [[ATTR_NOUNWIND:#[0-9]+]]
define void @call_linkonceodr() {
    call void @linkonceodr()
    ret void
}

; CHECK: define dso_local void @call_weakodr() [[ATTR_NOUNWIND:#[0-9]+]]
define void @call_weakodr() {
    call void @weakodr()
    ret void
}

; CHECK: define dso_local void @call_linkonce() [[ATTR_NOUNWIND:#[0-9]+]]
define void @call_linkonce() {
    call void @linkonce()
    ret void
}

; CHECK: define dso_local void @call_weak() [[ATTR_NOUNWIND:#[0-9]+]]
define void @call_weak() {
    call void @weak()
    ret void
}

; CHECK: define dso_local void @call_linkonce_may_unwind() [[ATTR_NOUNWIND:#[0-9]+]]
;; The prevailing copy here comes from b.ll, which contains nounwind and norecurse
define void @call_linkonce_may_unwind() {
    call void @linkonce_may_unwind()
    ret void
}

; CHECK: define dso_local void @call_weak_may_unwind() [[ATTR_MAYTHROW:#[0-9]+]]
;; The prevailing copy hree comes from c.ll, which only contains norecurse
define void @call_weak_may_unwind() {
    call void @weak_may_unwind()
    ret void
}

; CHECK-DAG: attributes [[ATTR_NOUNWIND]] = { norecurse nounwind }
; CHECK-DAG: attributes [[ATTR_MAYTHROW]] = { norecurse }

;--- b.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

attributes #0 = { nounwind norecurse }

define void @extern() #0 {
  ret void
}

define linkonce_odr void @linkonceodr() #0 {
  ret void
}

define weak_odr void @weakodr() #0 {
  ret void
}

define linkonce void @linkonce() #0 {
  ret void
}

define weak void @weak() #0 {
  ret void
}

define linkonce void @linkonce_may_unwind() #0 {
  ret void
}

define weak void @weak_may_unwind() #0 {
  ret void
}

;--- c.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

attributes #0 = { nounwind norecurse }
attributes #1 = { norecurse }

define void @extern() #0 {
  ret void
}

define linkonce_odr void @linkonceodr() #0 {
  ret void
}

define weak_odr void @weakodr() #0 {
  ret void
}

define linkonce void @linkonce() #0 {
  ret void
}

define weak void @weak() #0 {
  ret void
}

declare void @may_throw()

define linkonce void @linkonce_may_unwind() #1 {
  call void @may_throw()
  ret void
}

define weak void @weak_may_unwind() #1 {
  call void @may_throw()
  ret void
}
