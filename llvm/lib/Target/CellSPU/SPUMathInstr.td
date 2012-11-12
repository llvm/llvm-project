//===-- SPUMathInst.td - Cell SPU math operations ---------*- tablegen -*--===//
//
//                     Cell SPU math operations
//
// This target description file contains instruction sequences for various
// math operations, such as vector multiplies, i32 multiply, etc., for the
// SPU's i32, i16 i8 and corresponding vector types.
//
// Any resemblance to libsimdmath or the Cell SDK simdmath library is
// purely and completely coincidental.
//===----------------------------------------------------------------------===//

//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
// v16i8 multiply instruction sequence:
//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

def : Pat<(mul (v16i8 VECREG:$rA), (v16i8 VECREG:$rB)),
          (ORv4i32
           (ANDv4i32
            (SELBv4i32 (MPYv8i16 VECREG:$rA, VECREG:$rB),
                       (SHLHIv8i16 (MPYv8i16 (ROTMAHIv8i16 VECREG:$rA, 8),
                                             (ROTMAHIv8i16 VECREG:$rB, 8)), 8),
                       (FSMBIv8i16 0x2222)),
            (ILAv4i32 0x0000ffff)),
           (SHLIv4i32
            (SELBv4i32 (MPYv8i16 (ROTMAIv4i32_i32 VECREG:$rA, 16),
                                 (ROTMAIv4i32_i32 VECREG:$rB, 16)),
                       (SHLHIv8i16 (MPYv8i16 (ROTMAIv4i32_i32 VECREG:$rA, 8),
                                             (ROTMAIv4i32_i32 VECREG:$rB, 8)), 8),
                       (FSMBIv8i16 0x2222)), 16))>;
                        
//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
// v8i16 multiply instruction sequence:
//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

def : Pat<(mul (v8i16 VECREG:$rA), (v8i16 VECREG:$rB)),
          (SELBv8i16 (MPYv8i16 VECREG:$rA, VECREG:$rB),
                     (SHLIv4i32 (MPYHHv8i16 VECREG:$rA, VECREG:$rB), 16),
                     (FSMBIv8i16 0xcccc))>;
                 
//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
// v4i32, i32 multiply instruction sequence:
//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

def MPYv4i32:
  Pat<(mul (v4i32 VECREG:$rA), (v4i32 VECREG:$rB)),
      (Av4i32
        (v4i32 (Av4i32 (v4i32 (MPYHv4i32 VECREG:$rA, VECREG:$rB)),
                       (v4i32 (MPYHv4i32 VECREG:$rB, VECREG:$rA)))),
        (v4i32 (MPYUv4i32 VECREG:$rA, VECREG:$rB)))>;

def MPYi32:
  Pat<(mul R32C:$rA, R32C:$rB),
      (Ar32
        (Ar32 (MPYHr32 R32C:$rA, R32C:$rB),
              (MPYHr32 R32C:$rB, R32C:$rA)),
        (MPYUr32 R32C:$rA, R32C:$rB))>;

//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
// f32, v4f32 divide instruction sequence:
//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

// Reciprocal estimate and interpolation
def Interpf32: CodeFrag<(FIf32 R32FP:$rB, (FRESTf32 R32FP:$rB))>;
// Division estimate
def DivEstf32: CodeFrag<(FMf32 R32FP:$rA, Interpf32.Fragment)>;
// Newton-Raphson iteration
def NRaphf32: CodeFrag<(FMAf32 (FNMSf32 DivEstf32.Fragment, R32FP:$rB, R32FP:$rA),
                               Interpf32.Fragment,
                               DivEstf32.Fragment)>;
// Epsilon addition
def Epsilonf32: CodeFrag<(AIf32 NRaphf32.Fragment, 1)>;

def : Pat<(fdiv R32FP:$rA, R32FP:$rB),
          (SELBf32_cond NRaphf32.Fragment,
                        Epsilonf32.Fragment,
                        (CGTIf32 (FNMSf32 R32FP:$rB, Epsilonf32.Fragment, R32FP:$rA), -1))>;

// Reciprocal estimate and interpolation
def Interpv4f32: CodeFrag<(FIv4f32 (v4f32 VECREG:$rB), (FRESTv4f32 (v4f32 VECREG:$rB)))>;
// Division estimate
def DivEstv4f32: CodeFrag<(FMv4f32 (v4f32 VECREG:$rA), Interpv4f32.Fragment)>;
// Newton-Raphson iteration
def NRaphv4f32: CodeFrag<(FMAv4f32 (FNMSv4f32 DivEstv4f32.Fragment,
                                              (v4f32 VECREG:$rB),
                                              (v4f32 VECREG:$rA)),
                                   Interpv4f32.Fragment,
                                   DivEstv4f32.Fragment)>;
// Epsilon addition
def Epsilonv4f32: CodeFrag<(AIv4f32 NRaphv4f32.Fragment, 1)>;

def : Pat<(fdiv (v4f32 VECREG:$rA), (v4f32 VECREG:$rB)),
          (SELBv4f32_cond NRaphv4f32.Fragment,
                        Epsilonv4f32.Fragment,
                        (CGTIv4f32 (FNMSv4f32 (v4f32 VECREG:$rB),
                                              Epsilonv4f32.Fragment,
                                              (v4f32 VECREG:$rA)), -1))>;
