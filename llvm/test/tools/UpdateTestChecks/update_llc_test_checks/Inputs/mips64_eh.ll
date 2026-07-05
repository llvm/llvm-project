; RUN: llc -mtriple=mips64-unknown-linux < %s | FileCheck %s

define i32 @main() personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*){
  %1 = invoke i32 @foo() to label %good unwind label %bad
good:
  ret i32 5
bad:
  %2 = landingpad { i8*, i32 }
  cleanup
  resume { i8*, i32 } %2
}

declare i32 @foo()
declare i32 @__gxx_personality_v0(...)
