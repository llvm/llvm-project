; RUN: llc < %s

; Make sure we are not crashing on this one.

target triple = "x86_64-unknown-linux-gnu"

%"Iterator" = type { i32* }

declare { i64, <2 x float> } @Call() 
declare { i64, <2 x float> }* @CallPtr() 

define { i64, <2 x float> } @Foo(%"Iterator"* %this) {
entry:
  %retval = alloca i32
  %this.addr = alloca %"Iterator"*
  %this1 = load %"Iterator"** %this.addr
  %bundle_ = getelementptr inbounds %"Iterator"* %this1, i32 0, i32 0
  %0 = load i32** %bundle_
  %1 = call { i64, <2 x float> } @Call()
  %2 = call { i64, <2 x float> }* @CallPtr()
  %3 = getelementptr { i64, <2 x float> }* %2, i32 0, i32 1
  %4 = extractvalue { i64, <2 x float> } %1, 1
  store <2 x float> %4, <2 x float>* %3
  %5 = load { i64, <2 x float> }* %2
  ret { i64, <2 x float> } %5
}

