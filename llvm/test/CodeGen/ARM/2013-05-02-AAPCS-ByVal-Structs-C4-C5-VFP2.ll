;Check AAPCS, 5.5 Parameters Passing, C4 and C5 rules.
;Check case when NSAA != 0, and NCRN < R4, NCRN+ParamSize > R4
;RUN: llc -mtriple=thumbv7-linux-gnueabihf -float-abi=hard < %s | FileCheck %s

%st_t = type { i32, i32, i32, i32 }
@static_val = constant %st_t { i32 777, i32 888, i32 787, i32 878}

define void @foo(double %vfp0,     ; --> D0,              NSAA=SP
                 double %vfp1,     ; --> D1,              NSAA=SP
		 double %vfp2,     ; --> D2,              NSAA=SP
		 double %vfp3,     ; --> D3,              NSAA=SP
		 double %vfp4,     ; --> D4,              NSAA=SP
		 double %vfp5,     ; --> D5,              NSAA=SP
		 double %vfp6,     ; --> D6,              NSAA=SP
		 double %vfp7,     ; --> D7,              NSAA=SP
		 double %vfp8,     ; --> SP,              NSAA=SP+8 (!)
                 i32 %p0,          ; --> R0,              NSAA=SP+8
		 %st_t* byval(%st_t) %p1, ; --> SP+8, 4 words    NSAA=SP+24
		 i32 %p2) #0 {     ; --> SP+24,           NSAA=SP+24

entry:
  ;CHECK:  push {r7, lr}
  ;CHECK:  ldr    r0, [sp, #32]
  ;CHECK:  bl     fooUseI32
  call void @fooUseI32(i32 %p2)
  ret void
}

declare void @fooUseI32(i32)

define void @doFoo() {
entry:
  call void @foo(double 23.0,
                 double 23.1,
                 double 23.2,
                 double 23.3,
                 double 23.4,
                 double 23.5,
                 double 23.6,
                 double 23.7,
                 double 23.8,
                 i32 0, %st_t* byval(%st_t) @static_val, i32 1)
  ret void
}

