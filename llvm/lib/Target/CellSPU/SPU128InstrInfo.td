//===-- SPU128InstrInfo.td - Cell SPU 128-bit operations --*- tablegen -*--===//
//
//                     Cell SPU 128-bit operations
//
//===----------------------------------------------------------------------===//

// zext 32->128: Zero extend 32-bit to 128-bit
def : Pat<(i128 (zext R32C:$rSrc)),
          (ROTQMBYIr128_zext_r32 R32C:$rSrc, 12)>;

// zext 64->128: Zero extend 64-bit to 128-bit
def : Pat<(i128 (zext R64C:$rSrc)),
          (ROTQMBYIr128_zext_r64 R64C:$rSrc, 8)>;

// zext 16->128: Zero extend 16-bit to 128-bit
def : Pat<(i128 (zext R16C:$rSrc)),
          (ROTQMBYIr128_zext_r32 (ANDi16i32 R16C:$rSrc, (ILAr32 0xffff)), 12)>;

// zext 8->128: Zero extend 8-bit to 128-bit
def : Pat<(i128 (zext R8C:$rSrc)),
          (ROTQMBYIr128_zext_r32 (ANDIi8i32 R8C:$rSrc, 0xf), 12)>;

// anyext 32->128: Zero extend 32-bit to 128-bit
def : Pat<(i128 (anyext R32C:$rSrc)),
          (ROTQMBYIr128_zext_r32 R32C:$rSrc, 12)>;

// anyext 64->128: Zero extend 64-bit to 128-bit
def : Pat<(i128 (anyext R64C:$rSrc)),
          (ROTQMBYIr128_zext_r64 R64C:$rSrc, 8)>;

// anyext 16->128: Zero extend 16-bit to 128-bit
def : Pat<(i128 (anyext R16C:$rSrc)),
          (ROTQMBYIr128_zext_r32 (ANDi16i32 R16C:$rSrc, (ILAr32 0xffff)), 12)>;

// anyext 8->128: Zero extend 8-bit to 128-bit
def : Pat<(i128 (anyext R8C:$rSrc)),
          (ROTQMBYIr128_zext_r32 (ANDIi8i32 R8C:$rSrc, 0xf), 12)>;

// Shift left
def : Pat<(shl GPRC:$rA, R32C:$rB),
          (SHLQBYBIr128 (SHLQBIr128 GPRC:$rA, R32C:$rB), R32C:$rB)>;
