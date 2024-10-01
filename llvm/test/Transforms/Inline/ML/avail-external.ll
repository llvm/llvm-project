; REQUIRES: x86_64-linux
; RUN: rm -rf %t.rundir
; RUN: rm -rf %t.channel-basename.*
; RUN: mkdir %t.rundir
; RUN: cp %S/../../../../lib/Analysis/models/log_reader.py %t.rundir
; RUN: cp %S/../../../../lib/Analysis/models/interactive_host.py %t.rundir
; RUN: cp %S/Inputs/interactive_main.py %t.rundir
; RUN: %python %t.rundir/interactive_main.py %t.channel-basename \
; RUN:    opt -passes=scc-oz-module-inliner -interactive-model-runner-echo-reply \
; RUN:    -enable-ml-inliner=release -inliner-interactive-channel-base=%t.channel-basename %s -S -o /dev/null | FileCheck %s

define available_externally void @g() {
  ret void
}

define void @f(){
  call void @g()
  ret void
}

; CHECK: is_callee_avail_external: 1
; CHECK: is_caller_avail_external: 0
