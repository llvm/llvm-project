;Check 5.5 Parameter Passing --> Stage C --> C.4 statement, when NSAA is not
;equal to SP.
;
; Our purpose: make NSAA != SP, and only after start to use GPRs. 
;
;Co-Processor register candidates may be either in VFP or in stack, so after
;all VFP are allocated, stack is used. We can use stack without GPR allocation
;in that case, passing 9 f64 params, for example.
;First eight params goes to d0-d7, ninth one goes to the stack.
;Now, as 10th parameter, we pass i32, and it must go to R0.
;
;5.5 Parameter Passing, Stage C:
;
;C.2.cp If the argument is a CPRC then any co-processor registers in that class
;that are unallocated are marked as unavailable. The NSAA is adjusted upwards
;until it is correctly aligned for the argument and the argument is copied to
;the memory at the adjusted NSAA. The NSAA is further incremented by the size
;of the argument. The argument has now been allocated.
;...
;C.4 If the size in words of the argument is not more than r4 minus NCRN, the
;argument is copied into core registers, starting at the NCRN. The NCRN is
;incremented by the number of registers used. Successive registers hold the
;parts of the argument they would hold if its value were loaded into those
;registers from memory using an LDM instruction. The argument has now been
;allocated.
;
;What is actually checked here:
;Here we check that i32 param goes to r0.
;
;Current test-case was produced with command:
;arm-linux-gnueabihf-clang -mcpu=cortex-a9 params-to-GPR.c -S -O1 -emit-llvm
;
;// params-to-GRP.c:
;
;void fooUseI32(unsigned);
;
;void foo(long double p0,
;         long double p1,
;         long double p2,
;         long double p3,
;         long double p4,
;         long double p5,
;         long double p6,
;         long double p7,
;         long double p8,
;         unsigned p9) {
;  fooUseI32(p9);
;}
;
;void doFoo() {
;  foo( 1,2,3,4,5,6,7,8,9, 43 );
;}

;RUN: llc -mtriple=thumbv7-linux-gnueabihf -float-abi=hard < %s | FileCheck %s
;
;CHECK-LABEL:     foo:
;CHECK-NOT:     mov r0
;CHECK-NOT:     ldr r0
;CHECK:         bl fooUseI32
;CHECK-LABEL:     doFoo:
;CHECK:         movs    r0, #43
;CHECK:         bl      foo

define void @foo(double %p0, ; --> D0
                 double %p1, ; --> D1
		 double %p2, ; --> D2
		 double %p3, ; --> D3
		 double %p4, ; --> D4
		 double %p5, ; --> D5
		 double %p6, ; --> D6
		 double %p7, ; --> D7
		 double %p8, ; --> Stack
		 i32 %p9) #0 { ; --> R0, not Stack+8
entry:
  call void @fooUseI32(i32 %p9)
  ret void
}

declare void @fooUseI32(i32)

define void @doFoo() {
entry:
  tail call void @foo(double 23.0, ; --> D0
                      double 23.1, ; --> D1
		      double 23.2, ; --> D2
                      double 23.3, ; --> D3
                      double 23.4, ; --> D4
                      double 23.5, ; --> D5
                      double 23.6, ; --> D6
                      double 23.7, ; --> D7
                      double 23.8, ; --> Stack
                      i32 43)      ; --> R0, not Stack+8
  ret void
}

