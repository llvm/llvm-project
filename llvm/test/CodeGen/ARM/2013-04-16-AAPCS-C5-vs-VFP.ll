;Check 5.5 Parameter Passing --> Stage C --> C.5 statement, when NSAA is not
;equal to SP.
;
; Our purpose: make NSAA != SP, and only after start to use GPRs, then pass
;              byval parameter and check that it goes to stack only.
;
;Co-Processor register candidates may be either in VFP or in stack, so after
;all VFP are allocated, stack is used. We can use stack without GPR allocation
;in that case, passing 9 f64 params, for example.
;First eight params goes to d0-d7, ninth one goes to the stack.
;Now, as 10th parameter, we pass i32, and it must go to R0.
;
;For more information,
;please, read 5.5 Parameter Passing, Stage C, stages C.2.cp, C.4 and C.5
;
;
;RUN: llc -mtriple=thumbv7-linux-gnueabihf -float-abi=hard < %s | FileCheck %s

%struct_t = type { i32, i32, i32, i32 }
@static_val = constant %struct_t { i32 777, i32 888, i32 999, i32 1000 }
declare void @fooUseStruct(%struct_t*)

define void @foo2(double %p0, ; --> D0
                  double %p1, ; --> D1
		  double %p2, ; --> D2
		  double %p3, ; --> D3
		  double %p4, ; --> D4
		  double %p5, ; --> D5
		  double %p6, ; --> D6
		  double %p7, ; --> D7
		  double %p8, ; --> Stack
		  i32 %p9,    ; --> R0
                  %struct_t* byval(%struct_t) %p10) ; --> Stack+8
{
entry:
;CHECK:     push {r7, lr}
;CHECK-NOT: stm
;CHECK:     add r0, sp, #16
;CHECK:     bl fooUseStruct
  call void @fooUseStruct(%struct_t* %p10)

  ret void
}

define void @doFoo2() {
entry:
;CHECK-NOT: ldm
  tail call void @foo2(double 23.0, ; --> D0
                       double 23.1, ; --> D1
		       double 23.2, ; --> D2
                       double 23.3, ; --> D3
                       double 23.4, ; --> D4
                       double 23.5, ; --> D5
                       double 23.6, ; --> D6
                       double 23.7, ; --> D7
                       double 23.8, ; --> Stack
                       i32 43,      ; --> R0, not Stack+8
                       %struct_t* byval(%struct_t) @static_val) ; --> Stack+8, not R1
  ret void
}

