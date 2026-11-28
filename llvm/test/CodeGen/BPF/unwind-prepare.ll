; RUN: llc -mtriple=bpfel -mcpu=v4 -print-after=dwarf-eh-prepare -filetype=null \
; RUN:   %s 2>&1 | FileCheck %s

; resume is llvm.bpf.unwind.exit before DwarfEHPrepare runs, so it has no
; resume to turn into a call of _Unwind_Resume().

declare void @may_throw()
declare void @bpf_free(ptr)
declare i32 @rust_eh_personality(i32, i64, ptr, ptr)

define void @cleanup(ptr %node) uwtable personality ptr @rust_eh_personality {
entry:
  invoke void @llvm.bpf.unwind.exit()
          to label %failed unwind label %cleanup

failed:
  unreachable

cleanup:
  %lp = landingpad { ptr, i32 } cleanup
  call void @bpf_free(ptr %node) nounwind
  resume { ptr, i32 } %lp
}

; CHECK-LABEL: define void @cleanup(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    invoke void @llvm.bpf.unwind.exit()
; CHECK-NEXT:            to label %failed unwind label %cleanup{{$}}
; CHECK-EMPTY:
; CHECK-NEXT:  failed:
; CHECK-NEXT:    unreachable{{$}}
; CHECK-EMPTY:
; CHECK-NEXT:  cleanup:
; CHECK-NEXT:    %lp = landingpad { ptr, i32 }
; CHECK-NEXT:            cleanup{{$}}
; CHECK-NEXT:    call void @bpf_free(ptr %node) #{{[0-9]+}}{{$}}
; CHECK-NEXT:    call void @llvm.bpf.unwind.exit(){{$}}
; CHECK-NEXT:    unreachable{{$}}
; CHECK-NEXT:  }

; A call that may unwind gets a pad that leaves the frame.
define void @passes_on() uwtable {
entry:
  call void @may_throw()
  ret void
}

; CHECK-LABEL: define void @passes_on(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    invoke void @may_throw()
; CHECK-NEXT:            to label %.noexc unwind label %unwind.resume{{$}}
; CHECK-EMPTY:
; CHECK-NEXT:  .noexc:
; CHECK-NEXT:    ret void{{$}}
; CHECK-EMPTY:
; CHECK-NEXT:  unwind.resume:
; CHECK-NEXT:    %0 = landingpad { ptr, i32 }
; CHECK-NEXT:            cleanup{{$}}
; CHECK-NEXT:    call void @llvm.bpf.unwind.exit(){{$}}
; CHECK-NEXT:    unreachable{{$}}
; CHECK-NEXT:  }
