; RUN: llc -mtriple=x86_64-w64-windows-gnu < %s | FileCheck %s --check-prefix=GNU
; RUN: llc -mtriple=x86_64-w64-windows-gnu < %s -filetype=obj | llvm-objdump - --headers | FileCheck %s --check-prefix=GNUOBJ

; When doing GCC style comdats for MinGW, the .xdata sections don't have a normal comdat
; symbol attached, which requires a bit of adjustments for the assembler output.

; Generated with this C++ source:
; int bar(int);
; __declspec(selectany) int gv = 42;
; inline int foo(int x) { try { return bar(x) + gv; } catch (...) { return 0; } }
; int main() { return foo(1); }

$_Z3fooi = comdat any

$gv = comdat any

@gv = weak_odr dso_local global i32 42, comdat, align 4

; Function Attrs: norecurse uwtable
define dso_local i32 @main() #0 {
entry:
  %call = tail call i32 @_Z3fooi(i32 1)
  ret i32 %call
}

; GNU: main:

; Function Attrs: inlinehint uwtable
define linkonce_odr dso_local i32 @_Z3fooi(i32 %x) #1 comdat personality i8* bitcast (i32 (...)* @__gxx_personality_seh0 to i8*) {
entry:
  %call = invoke i32 @_Z3bari(i32 %x)
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %entry
  %0 = load i32, i32* @gv, align 4
  %add = add nsw i32 %0, %call
  br label %return

lpad:                                             ; preds = %entry
  %1 = landingpad { i8*, i32 }
          catch i8* null
  %2 = extractvalue { i8*, i32 } %1, 0
  %3 = tail call i8* @__cxa_begin_catch(i8* %2) #3
  tail call void @__cxa_end_catch()
  br label %return

return:                                           ; preds = %lpad, %invoke.cont
  %retval.0 = phi i32 [ %add, %invoke.cont ], [ 0, %lpad ]
  ret i32 %retval.0
}

; The .xdata section below doesn't have the usual comdat symbol attached, which requires
; a different syntax for the assembly output.

; GNU: .section        .text$_Z3fooi,"xr",discard,_Z3fooi
; GNU: _Z3fooi:
; GNU: .section        .xdata$_Z3fooi,"dr"
; GNU: .linkonce       discard
; GNU: GCC_except_table1:
; GNU: .section        .data$gv,"dw",discard,gv
; GNU: gv:
; GNU: .long 42

; Make sure the assembler puts the .xdata and .pdata in sections with the right
; names.
; GNUOBJ: .text$_Z3fooi
; GNUOBJ: .xdata$_Z3fooi
; GNUOBJ: .data$gv
; GNUOBJ: .pdata$_Z3fooi

declare dso_local i32 @_Z3bari(i32)

declare dso_local i32 @__gxx_personality_seh0(...)

declare dso_local i8* @__cxa_begin_catch(i8*) local_unnamed_addr

declare dso_local void @__cxa_end_catch() local_unnamed_addr

attributes #0 = { norecurse uwtable }
attributes #1 = { inlinehint uwtable }
attributes #3 = { nounwind }
