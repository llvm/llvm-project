# RUN: llvm-mc --verify -triple riscv32 -mattr=+zfa,+zfh -M no-aliases -show-encoding < %s
# RUN: llvm-mc --verify -triple riscv64 -mattr=+zfa,+zfh -M no-aliases -show-encoding < %s

fminm.q fa0, fa1, fa2 # expected-error {{instruction requires the following: 'Q' (Quad-Precision Floating-Point)}}

fmaxm.q fs3, fs4, fs5 # expected-error {{instruction requires the following: 'Q' (Quad-Precision Floating-Point)}}

fround.q fs1, fs2 # expected-error {{instruction requires the following: 'Q' (Quad-Precision Floating-Point)}}

fround.q fs1, fs2, dyn # expected-error {{instruction requires the following: 'Q' (Quad-Precision Floating-Point)}}

fround.q fs1, fs2, rtz # expected-error {{instruction requires the following: 'Q' (Quad-Precision Floating-Point)}}

fround.q fs1, fs2, rne # expected-error {{instruction requires the following: 'Q' (Quad-Precision Floating-Point)}}

froundnx.q fs1, fs2 # expected-error {{instruction requires the following: 'Q' (Quad-Precision Floating-Point)}}

froundnx.q fs1, fs2, dyn # expected-error {{instruction requires the following: 'Q' (Quad-Precision Floating-Point)}}

froundnx.q fs1, fs2, rtz # expected-error {{instruction requires the following: 'Q' (Quad-Precision Floating-Point)}}

froundnx.q fs1, fs2, rne # expected-error {{instruction requires the following: 'Q' (Quad-Precision Floating-Point)}}

fltq.q a1, fs1, fs2 # expected-error {{instruction requires the following: 'Q' (Quad-Precision Floating-Point)}}

fleq.q a1, ft1, ft2 # expected-error {{instruction requires the following: 'Q' (Quad-Precision Floating-Point)}}
