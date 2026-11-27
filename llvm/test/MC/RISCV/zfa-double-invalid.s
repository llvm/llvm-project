# RUN: llvm-mc --verify -triple riscv32 -mattr=+zfa,+zfh -M no-aliases -show-encoding < %s
# RUN: llvm-mc --verify -triple riscv64 -mattr=+zfa,+zfh -M no-aliases -show-encoding < %s

fminm.d fa0, fa1, fa2 # expected-error {{instruction requires the following: 'D' (Double-Precision Floating-Point)}}

fmaxm.d fs3, fs4, fs5 # expected-error {{instruction requires the following: 'D' (Double-Precision Floating-Point)}}

fround.d fs1, fs2 # expected-error {{instruction requires the following: 'D' (Double-Precision Floating-Point)}}

fround.d fs1, fs2, dyn # expected-error {{instruction requires the following: 'D' (Double-Precision Floating-Point)}}

fround.d fs1, fs2, rtz # expected-error {{instruction requires the following: 'D' (Double-Precision Floating-Point)}}

fround.d fs1, fs2, rne # expected-error {{instruction requires the following: 'D' (Double-Precision Floating-Point)}}

froundnx.d fs1, fs2 # expected-error {{instruction requires the following: 'D' (Double-Precision Floating-Point)}}

froundnx.d fs1, fs2, dyn # expected-error {{instruction requires the following: 'D' (Double-Precision Floating-Point)}}

froundnx.d fs1, fs2, rtz # expected-error {{instruction requires the following: 'D' (Double-Precision Floating-Point)}}

froundnx.d fs1, fs2, rne # expected-error {{instruction requires the following: 'D' (Double-Precision Floating-Point)}}

fcvtmod.w.d a1, ft1, rtz # expected-error {{instruction requires the following: 'D' (Double-Precision Floating-Point)}}

fltq.d a1, fs1, fs2 # expected-error {{instruction requires the following: 'D' (Double-Precision Floating-Point)}}

fleq.d a1, ft1, ft2 # expected-error {{instruction requires the following: 'D' (Double-Precision Floating-Point)}}
