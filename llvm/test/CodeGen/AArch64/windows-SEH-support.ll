; RUN: llc -mtriple aarch64-unknown-windows-msvc -filetype asm -o /dev/null %s

declare dllimport void @f() local_unnamed_addr

declare dso_local i32 @__C_specific_handler(...)

define hidden swiftcc void @g() unnamed_addr personality i8* bitcast (i32 (...)* @__C_specific_handler to i8*) {
entry:
  invoke void @f() to label %__try.cont unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %0 = catchswitch within none [label %__except] unwind to caller

__except:
  %1 = catchpad within %0 [i8* null]              ; preds = %catch.dispatch
  catchret from %1 to label %__try.cont

__try.cont:                                       ; preds = %__except, %entry
  ret void
}

define hidden fastcc void @h() unnamed_addr personality i8* bitcast (i32 (...)* @__C_specific_handler to i8*) {
entry:
  invoke void @f() to label %__try.cont unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %0 = catchswitch within none [label %__except] unwind to caller

__except:                                         ; preds = %catch.dispatch
  %1 = catchpad within %0 [i8* null]
  catchret from %1 to label %__try.cont

__try.cont:                                       ; preds = %__except, %entry
  ret void
}

