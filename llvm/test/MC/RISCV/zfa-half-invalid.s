# RUN: llvm-mc --verify -triple riscv32 -mattr=+zfa,+d -M no-aliases -show-encoding < %s
# RUN: llvm-mc --verify -triple riscv64 -mattr=+zfa,+d -M no-aliases -show-encoding < %s

fminm.h fa0, fa1, fa2 # expected-error {{instruction requires the following: 'Zfh' (Half-Precision Floating-Point)}}

fmaxm.h fs3, fs4, fs5 # expected-error {{instruction requires the following: 'Zfh' (Half-Precision Floating-Point)}}

fround.h fs1, fs2 # expected-error {{instruction requires the following: 'Zfh' (Half-Precision Floating-Point)}}

fround.h fs1, fs2, dyn # expected-error {{instruction requires the following: 'Zfh' (Half-Precision Floating-Point)}}

fround.h fs1, fs2, rtz # expected-error {{instruction requires the following: 'Zfh' (Half-Precision Floating-Point)}}

fround.h fs1, fs2, rne # expected-error {{instruction requires the following: 'Zfh' (Half-Precision Floating-Point)}}

froundnx.h fs1, fs2 # expected-error {{instruction requires the following: 'Zfh' (Half-Precision Floating-Point)}}

froundnx.h fs1, fs2, dyn # expected-error {{instruction requires the following: 'Zfh' (Half-Precision Floating-Point)}}

froundnx.h fs1, fs2, rtz # expected-error {{instruction requires the following: 'Zfh' (Half-Precision Floating-Point)}}

froundnx.h fs1, fs2, rne # expected-error {{instruction requires the following: 'Zfh' (Half-Precision Floating-Point)}}

fltq.h a1, fs1, fs2 # expected-error {{instruction requires the following: 'Zfh' (Half-Precision Floating-Point)}}

fleq.h a1, ft1, ft2 # expected-error {{instruction requires the following: 'Zfh' (Half-Precision Floating-Point)}}
