; RUN: llc < %s

; Make sure we are not crashing on this one.

target triple = "x86_64-unknown-linux-gnu"

%"Iterator" = type { ptr }

declare { i64, <2 x float> } @Call() 
declare ptr @CallPtr() 

define { i64, <2 x float> } @Foo(ptr %this) {
entry:
  %retval = alloca i32
  %this.addr = alloca ptr
  %this1 = load ptr, ptr %this.addr
  %0 = load ptr, ptr %this1
  %1 = call { i64, <2 x float> } @Call()
  %2 = call ptr @CallPtr()
  %3 = getelementptr { i64, <2 x float> }, ptr %2, i32 0, i32 1
  %4 = extractvalue { i64, <2 x float> } %1, 1
  store <2 x float> %4, ptr %3
  %5 = load { i64, <2 x float> }, ptr %2
  ret { i64, <2 x float> } %5
}

