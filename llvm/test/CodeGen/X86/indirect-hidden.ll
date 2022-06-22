; RUN: llc -mtriple=i686-apple-macosx -o - %s | FileCheck %s

; x86 doesn't normally use indirect symbols, particularly hidden ones, but it
; can be tricked into it for exception-handling typeids.

@hidden_typeid = external hidden constant ptr
@normal_typeid = external constant ptr

declare void @throws()

define void @get_indirect_hidden() personality ptr @__gxx_personality_v0 {
  invoke void @throws() to label %end unwind label %lpad
lpad:
  %tmp = landingpad { ptr, i32 }
          catch ptr @hidden_typeid
  br label %end

end:
  ret void
}

define void @get_indirect() personality ptr @__gxx_personality_v0 {
  invoke void @throws() to label %end unwind label %lpad
lpad:
  %tmp = landingpad { ptr, i32 }
          catch ptr @normal_typeid
  br label %end

end:
  ret void
}

declare i32 @__gxx_personality_v0(...)

; CHECK: .section __IMPORT,__pointers,non_lazy_symbol_pointers

; CHECK-NOT: __DATA,__data
; CHECK: .indirect_symbol _hidden_typeid
; CHECK-NEXT: .long 0

; CHECK-NOT: __DATA,__data
; CHECK: .indirect_symbol _normal_typeid
; CHECK-NEXT: .long 0
