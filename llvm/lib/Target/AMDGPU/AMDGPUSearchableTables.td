//===-- AMDGPUSearchableTables.td - ------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Resource intrinsics table.
//===----------------------------------------------------------------------===//

class RsrcIntrinsic<AMDGPURsrcIntrinsic intr> {
  Intrinsic Intr = !cast<Intrinsic>(intr);
  bits<8> RsrcArg = intr.RsrcArg;
  bit IsImage = intr.IsImage;
}

def RsrcIntrinsics : GenericTable {
  let FilterClass = "RsrcIntrinsic";
  let Fields = ["Intr", "RsrcArg", "IsImage"];

  let PrimaryKey = ["Intr"];
  let PrimaryKeyName = "lookupRsrcIntrinsic";
}

foreach intr = !listconcat(AMDGPUBufferIntrinsics,
                           AMDGPUImageDimIntrinsics,
                           AMDGPUImageDimAtomicIntrinsics) in {
  def : RsrcIntrinsic<!cast<AMDGPURsrcIntrinsic>(intr)>;
}

//===----------------------------------------------------------------------===//
// Buffer intrinsic argument layout table.
//===----------------------------------------------------------------------===//

def BufferIntrinsicKind : GenericEnum {
  let FilterClass = "AMDGPUBufferIntrinsicKind";
  let NameField = "KindName";
  let ValueField = "Value";
  let UnderlyingType = "uint8_t";
}

class BufferIntrinsicInfo<AMDGPUBufferIntrinsic intr> {
  Intrinsic Intr = !cast<Intrinsic>(intr);
  AMDGPUBufferIntrinsicKind Kind = intr.BufferKind;
  bits<8> RsrcArg = intr.RsrcArg;
  bits<8> VDataArg = intr.VDataArg;
  bits<8> SizeArg = intr.SizeArg;
  bits<8> VIndexArg = intr.VIndexArg;
  bits<8> VOffsetArg = intr.VOffsetArg;
  bits<8> SOffsetArg = intr.SOffsetArg;
  bits<8> ImmOffsetArg = intr.ImmOffsetArg;
  bits<8> FormatArg = intr.FormatArg;
  bits<8> AuxArg = intr.AuxArg;
  bit IsFormatFromRsrc = intr.IsFormatFromRsrc;
  bit IsAsync = intr.IsAsync;

  // Check the computed layout against the intrinsic's signature. Absent arguments (255) are looked up in
  // a trailing i32 sentinel so that the checks don't need to special-case them.
  defvar Name = !cast<string>(Intr);
  defvar Params = Intr.ParamTypes;
  defvar PaddedParams = !listconcat(Params, [llvm_i32_ty]);
  defvar ArgIsI32 = !foldl(true, [intr.VIndexArg, intr.VOffsetArg, intr.SOffsetArg, intr.ImmOffsetArg, intr.FormatArg, intr.AuxArg, intr.SizeArg],
                           acc, idx, !and(acc, !eq(PaddedParams[!if(!eq(idx, 255), !size(Params), idx)], llvm_i32_ty)));
  defvar LDSPtrIdx = !if(!eq(Kind, AMDGPUBufferKindLoadToLDS), !add(intr.RsrcArg, 1), !size(Params));
  assert !eq(!size(Params), intr.NumArgs),
    "buffer intrinsic layout of " # Name # " has " # intr.NumArgs # " arguments, but the intrinsic has " # !size(Params);
  assert !or(!eq(Params[intr.RsrcArg], llvm_v4i32_ty), !eq(Params[intr.RsrcArg], AMDGPUBufferRsrcTy)),
    "rsrc argument of " # Name # " is not at position " # intr.RsrcArg;
  assert ArgIsI32, "offset/index/size/aux arguments of " # Name # " are not all i32";
  // There's code relying on this invariant, so we verify it. Didn't seem worth a table slot.
  assert !or(!eq(LDSPtrIdx, !size(Params)), !isa<LLVMQualPointerType>(PaddedParams[LDSPtrIdx])),
    "LDS pointer of " # Name # " does not follow the rsrc";
}

def BufferIntrinsicsTable : GenericTable {
  let FilterClass = "BufferIntrinsicInfo";
  let CppTypeName = "BufferIntrinsicInfo";
  let Fields = ["Intr", "Kind", "RsrcArg", "VDataArg", "SizeArg", "VIndexArg", "VOffsetArg", "SOffsetArg", "ImmOffsetArg", "FormatArg", "AuxArg", "IsFormatFromRsrc", "IsAsync"];
  string TypeOf_Kind = "BufferIntrinsicKind";

  let PrimaryKey = ["Intr"];
  let PrimaryKeyName = "lookupBufferIntrinsic";
}

foreach intr = AMDGPUBufferIntrinsics in {
  def : BufferIntrinsicInfo<intr>;
}

class GcnBufferFormatBase<bits<8> f, bits<8> bpc, bits<8> numc, bits<8> nfmt, bits<8> dfmt> {
  bits<8> Format = f;
  bits<8> BitsPerComp = bpc;
  bits<8> NumComponents = numc;
  bits<8> NumFormat = nfmt;
  bits<8> DataFormat = dfmt;
}

class Gfx9BufferFormat<bits<8> f, bits<8> bpc, bits<8> numc, bits<8> nfmt, bits<8> dfmt> : GcnBufferFormatBase<f, bpc, numc, nfmt, dfmt>;
class Gfx10BufferFormat<bits<8> f, bits<8> bpc, bits<8> numc, bits<8> nfmt, bits<8> dfmt> : GcnBufferFormatBase<f, bpc, numc, nfmt, dfmt>;
class Gfx11PlusBufferFormat<bits<8> f, bits<8> bpc, bits<8> numc, bits<8> nfmt, bits<8> dfmt> : GcnBufferFormatBase<f, bpc, numc, nfmt, dfmt>;

class GcnBufferFormatTable  : GenericTable {
  let CppTypeName = "GcnBufferFormatInfo";
  let Fields = ["Format", "BitsPerComp", "NumComponents", "NumFormat", "DataFormat"];
  let PrimaryKey = ["BitsPerComp", "NumComponents", "NumFormat"];
}

def Gfx9BufferFormat : GcnBufferFormatTable {
  let FilterClass = "Gfx9BufferFormat";
  let PrimaryKeyName = "getGfx9BufferFormatInfo";
}
def Gfx10BufferFormat : GcnBufferFormatTable {
  let FilterClass = "Gfx10BufferFormat";
  let PrimaryKeyName = "getGfx10BufferFormatInfo";
}
def Gfx11PlusBufferFormat : GcnBufferFormatTable {
  let FilterClass = "Gfx11PlusBufferFormat";
  let PrimaryKeyName = "getGfx11PlusBufferFormatInfo";
}

def getGfx9BufferFormatInfo : SearchIndex {
  let Table = Gfx9BufferFormat;
  let Key = ["Format"];
}
def getGfx10BufferFormatInfo : SearchIndex {
  let Table = Gfx10BufferFormat;
  let Key = ["Format"];
}
def getGfx11PlusBufferFormatInfo : SearchIndex {
  let Table = Gfx11PlusBufferFormat;
  let Key = ["Format"];
}

// Buffer formats with equal component sizes (GFX9 and earlier)
def : Gfx9BufferFormat< /*FORMAT_8_UNORM*/              0x01,  8, 1, /*NUM_FORMAT_UNORM*/   0, /*DATA_FORMAT_8*/            1>;
def : Gfx9BufferFormat< /*FORMAT_8_SNORM*/              0x11,  8, 1, /*NUM_FORMAT_SNORM*/   1, /*DATA_FORMAT_8*/            1>;
def : Gfx9BufferFormat< /*FORMAT_8_USCALED*/            0x21,  8, 1, /*NUM_FORMAT_USCALED*/ 2, /*DATA_FORMAT_8*/            1>;
def : Gfx9BufferFormat< /*FORMAT_8_SSCALED*/            0x31,  8, 1, /*NUM_FORMAT_SSCALED*/ 3, /*DATA_FORMAT_8*/            1>;
def : Gfx9BufferFormat< /*FORMAT_8_UINT*/               0x41,  8, 1, /*NUM_FORMAT_UINT*/    4, /*DATA_FORMAT_8*/            1>;
def : Gfx9BufferFormat< /*FORMAT_8_SINT*/               0x51,  8, 1, /*NUM_FORMAT_SINT*/    5, /*DATA_FORMAT_8*/            1>;
def : Gfx9BufferFormat< /*FORMAT_16_UNORM*/             0x02, 16, 1, /*NUM_FORMAT_UNORM*/   0, /*DATA_FORMAT_16*/           2>;
def : Gfx9BufferFormat< /*FORMAT_16_SNORM*/             0x12, 16, 1, /*NUM_FORMAT_SNORM*/   1, /*DATA_FORMAT_16*/           2>;
def : Gfx9BufferFormat< /*FORMAT_16_USCALED*/           0x22, 16, 1, /*NUM_FORMAT_USCALED*/ 2, /*DATA_FORMAT_16*/           2>;
def : Gfx9BufferFormat< /*FORMAT_16_SSCALED*/           0x32, 16, 1, /*NUM_FORMAT_SSCALED*/ 3, /*DATA_FORMAT_16*/           2>;
def : Gfx9BufferFormat< /*FORMAT_16_UINT*/              0x42, 16, 1, /*NUM_FORMAT_UINT*/    4, /*DATA_FORMAT_16*/           2>;
def : Gfx9BufferFormat< /*FORMAT_16_SINT*/              0x52, 16, 1, /*NUM_FORMAT_SINT*/    5, /*DATA_FORMAT_16*/           2>;
def : Gfx9BufferFormat< /*FORMAT_16_FLOAT*/             0x72, 16, 1, /*NUM_FORMAT_FLOAT*/   7, /*DATA_FORMAT_16*/           2>;
def : Gfx9BufferFormat< /*FORMAT_8_8_UNORM*/            0x03,  8, 2, /*NUM_FORMAT_UNORM*/   0, /*DATA_FORMAT_8_8*/          3>;
def : Gfx9BufferFormat< /*FORMAT_8_8_SNORM*/            0x13,  8, 2, /*NUM_FORMAT_SNORM*/   1, /*DATA_FORMAT_8_8*/          3>;
def : Gfx9BufferFormat< /*FORMAT_8_8_USCALED*/          0x23,  8, 2, /*NUM_FORMAT_USCALED*/ 2, /*DATA_FORMAT_8_8*/          3>;
def : Gfx9BufferFormat< /*FORMAT_8_8_SSCALED*/          0x33,  8, 2, /*NUM_FORMAT_SSCALED*/ 3, /*DATA_FORMAT_8_8*/          3>;
def : Gfx9BufferFormat< /*FORMAT_8_8_UINT*/             0x43,  8, 2, /*NUM_FORMAT_UINT*/    4, /*DATA_FORMAT_8_8*/          3>;
def : Gfx9BufferFormat< /*FORMAT_8_8_SINT*/             0x53,  8, 2, /*NUM_FORMAT_SINT*/    5, /*DATA_FORMAT_8_8*/          3>;
def : Gfx9BufferFormat< /*FORMAT_32_UINT*/              0x44, 32, 1, /*NUM_FORMAT_UINT*/    4, /*DATA_FORMAT_32*/           4>;
def : Gfx9BufferFormat< /*FORMAT_32_SINT*/              0x54, 32, 1, /*NUM_FORMAT_SINT*/    5, /*DATA_FORMAT_32*/           4>;
def : Gfx9BufferFormat< /*FORMAT_32_FLOAT*/             0x74, 32, 1, /*NUM_FORMAT_FLOAT*/   7, /*DATA_FORMAT_32*/           4>;
def : Gfx9BufferFormat< /*FORMAT_16_16_UNORM*/          0x05, 16, 2, /*NUM_FORMAT_UNORM*/   0, /*DATA_FORMAT_16_16*/        5>;
def : Gfx9BufferFormat< /*FORMAT_16_16_SNORM*/          0x15, 16, 2, /*NUM_FORMAT_SNORM*/   1, /*DATA_FORMAT_16_16*/        5>;
def : Gfx9BufferFormat< /*FORMAT_16_16_USCALED*/        0x25, 16, 2, /*NUM_FORMAT_USCALED*/ 2, /*DATA_FORMAT_16_16*/        5>;
def : Gfx9BufferFormat< /*FORMAT_16_16_SSCALED*/        0x35, 16, 2, /*NUM_FORMAT_SSCALED*/ 3, /*DATA_FORMAT_16_16*/        5>;
def : Gfx9BufferFormat< /*FORMAT_16_16_UINT*/           0x45, 16, 2, /*NUM_FORMAT_UINT*/    4, /*DATA_FORMAT_16_16*/        5>;
def : Gfx9BufferFormat< /*FORMAT_16_16_SINT*/           0x55, 16, 2, /*NUM_FORMAT_SINT*/    5, /*DATA_FORMAT_16_16*/        5>;
def : Gfx9BufferFormat< /*FORMAT_16_16_FLOAT*/          0x75, 16, 2, /*NUM_FORMAT_FLOAT*/   7, /*DATA_FORMAT_16_16*/        5>;
def : Gfx9BufferFormat< /*FORMAT_8_8_8_8_UNORM*/        0x0A,  8, 4, /*NUM_FORMAT_UNORM*/   0, /*DATA_FORMAT_8_8_8_8*/     10>;
def : Gfx9BufferFormat< /*FORMAT_8_8_8_8_SNORM*/        0x1A,  8, 4, /*NUM_FORMAT_SNORM*/   1, /*DATA_FORMAT_8_8_8_8*/     10>;
def : Gfx9BufferFormat< /*FORMAT_8_8_8_8_USCALED*/      0x2A,  8, 4, /*NUM_FORMAT_USCALED*/ 2, /*DATA_FORMAT_8_8_8_8*/     10>;
def : Gfx9BufferFormat< /*FORMAT_8_8_8_8_SSCALED*/      0x3A,  8, 4, /*NUM_FORMAT_SSCALED*/ 3, /*DATA_FORMAT_8_8_8_8*/     10>;
def : Gfx9BufferFormat< /*FORMAT_8_8_8_8_UINT*/         0x4A,  8, 4, /*NUM_FORMAT_UINT*/    4, /*DATA_FORMAT_8_8_8_8*/     10>;
def : Gfx9BufferFormat< /*FORMAT_8_8_8_8_SINT*/         0x5A,  8, 4, /*NUM_FORMAT_SINT*/    5, /*DATA_FORMAT_8_8_8_8*/     10>;
def : Gfx9BufferFormat< /*FORMAT_32_32_UINT*/           0x4B, 32, 2, /*NUM_FORMAT_UINT*/    4, /*DATA_FORMAT_32_32*/       11>;
def : Gfx9BufferFormat< /*FORMAT_32_32_SINT*/           0x5B, 32, 2, /*NUM_FORMAT_SINT*/    5, /*DATA_FORMAT_32_32*/       11>;
def : Gfx9BufferFormat< /*FORMAT_32_32_FLOAT*/          0x7B, 32, 2, /*NUM_FORMAT_FLOAT*/   7, /*DATA_FORMAT_32_32*/       11>;
def : Gfx9BufferFormat< /*FORMAT_16_16_16_16_UNORM*/    0x0C, 16, 4, /*NUM_FORMAT_UNORM*/   0, /*DATA_FORMAT_16_16_16_16*/ 12>;
def : Gfx9BufferFormat< /*FORMAT_16_16_16_16_SNORM*/    0x1C, 16, 4, /*NUM_FORMAT_SNORM*/   1, /*DATA_FORMAT_16_16_16_16*/ 12>;
def : Gfx9BufferFormat< /*FORMAT_16_16_16_16_USCALED*/  0x2C, 16, 4, /*NUM_FORMAT_USCALED*/ 2, /*DATA_FORMAT_16_16_16_16*/ 12>;
def : Gfx9BufferFormat< /*FORMAT_16_16_16_16_SSCALED*/  0x3C, 16, 4, /*NUM_FORMAT_SSCALED*/ 3, /*DATA_FORMAT_16_16_16_16*/ 12>;
def : Gfx9BufferFormat< /*FORMAT_16_16_16_16_UINT*/     0x4C, 16, 4, /*NUM_FORMAT_UINT*/    4, /*DATA_FORMAT_16_16_16_16*/ 12>;
def : Gfx9BufferFormat< /*FORMAT_16_16_16_16_SINT*/     0x5C, 16, 4, /*NUM_FORMAT_SINT*/    5, /*DATA_FORMAT_16_16_16_16*/ 12>;
def : Gfx9BufferFormat< /*FORMAT_16_16_16_16_FLOAT*/    0x7C, 16, 4, /*NUM_FORMAT_FLOAT*/   7, /*DATA_FORMAT_16_16_16_16*/ 12>;
def : Gfx9BufferFormat< /*FORMAT_32_32_32_UINT*/        0x4D, 32, 3, /*NUM_FORMAT_UINT*/    4, /*DATA_FORMAT_32_32_32*/    13>;
def : Gfx9BufferFormat< /*FORMAT_32_32_32_SINT*/        0x5D, 32, 3, /*NUM_FORMAT_SINT*/    5, /*DATA_FORMAT_32_32_32*/    13>;
def : Gfx9BufferFormat< /*FORMAT_32_32_32_FLOAT*/       0x7D, 32, 3, /*NUM_FORMAT_FLOAT*/   7, /*DATA_FORMAT_32_32_32*/    13>;
def : Gfx9BufferFormat< /*FORMAT_32_32_32_32_UINT*/     0x4E, 32, 4, /*NUM_FORMAT_UINT*/    4, /*DATA_FORMAT_32_32_32_32*/ 14>;
def : Gfx9BufferFormat< /*FORMAT_32_32_32_32_SINT*/     0x5E, 32, 4, /*NUM_FORMAT_SINT*/    5, /*DATA_FORMAT_32_32_32_32*/ 14>;
def : Gfx9BufferFormat< /*FORMAT_32_32_32_32_FLOAT*/    0x7E, 32, 4, /*NUM_FORMAT_FLOAT*/   7, /*DATA_FORMAT_32_32_32_32*/ 14>;

// Buffer formats with equal component sizes (GFX10 and later)
multiclass Gfx10PlusBufferFormat<bits<8> f, bits<8> bpc, bits<8> numc, bits<8> nfmt, bits<8> dfmt> {
  def : Gfx10BufferFormat<f, bpc, numc, nfmt, dfmt>;
  def : Gfx11PlusBufferFormat<f, bpc, numc, nfmt, dfmt>;
}
defm : Gfx10PlusBufferFormat< /*FORMAT_8_UNORM*/              0x01,  8, 1, /*NUM_FORMAT_UNORM*/   0, /*DATA_FORMAT_8*/            1>;
defm : Gfx10PlusBufferFormat< /*FORMAT_8_SNORM*/              0x02,  8, 1, /*NUM_FORMAT_SNORM*/   1, /*DATA_FORMAT_8*/            1>;
defm : Gfx10PlusBufferFormat< /*FORMAT_8_USCALED*/            0x03,  8, 1, /*NUM_FORMAT_USCALED*/ 2, /*DATA_FORMAT_8*/            1>;
defm : Gfx10PlusBufferFormat< /*FORMAT_8_SSCALED*/            0x04,  8, 1, /*NUM_FORMAT_SSCALED*/ 3, /*DATA_FORMAT_8*/            1>;
defm : Gfx10PlusBufferFormat< /*FORMAT_8_UINT*/               0x05,  8, 1, /*NUM_FORMAT_UINT*/    4, /*DATA_FORMAT_8*/            1>;
defm : Gfx10PlusBufferFormat< /*FORMAT_8_SINT*/               0x06,  8, 1, /*NUM_FORMAT_SINT*/    5, /*DATA_FORMAT_8*/            1>;
defm : Gfx10PlusBufferFormat< /*FORMAT_16_UNORM*/             0x07, 16, 1, /*NUM_FORMAT_UNORM*/   0, /*DATA_FORMAT_16*/           2>;
defm : Gfx10PlusBufferFormat< /*FORMAT_16_SNORM*/             0x08, 16, 1, /*NUM_FORMAT_SNORM*/   1, /*DATA_FORMAT_16*/           2>;
defm : Gfx10PlusBufferFormat< /*FORMAT_16_USCALED*/           0x09, 16, 1, /*NUM_FORMAT_USCALED*/ 2, /*DATA_FORMAT_16*/           2>;
defm : Gfx10PlusBufferFormat< /*FORMAT_16_SSCALED*/           0x0A, 16, 1, /*NUM_FORMAT_SSCALED*/ 3, /*DATA_FORMAT_16*/           2>;
defm : Gfx10PlusBufferFormat< /*FORMAT_16_UINT*/              0x0B, 16, 1, /*NUM_FORMAT_UINT*/    4, /*DATA_FORMAT_16*/           2>;
defm : Gfx10PlusBufferFormat< /*FORMAT_16_SINT*/              0x0C, 16, 1, /*NUM_FORMAT_SINT*/    5, /*DATA_FORMAT_16*/           2>;
defm : Gfx10PlusBufferFormat< /*FORMAT_16_FLOAT*/             0x0D, 16, 1, /*NUM_FORMAT_FLOAT*/   7, /*DATA_FORMAT_16*/           2>;
defm : Gfx10PlusBufferFormat< /*FORMAT_8_8_UNORM*/            0x0E,  8, 2, /*NUM_FORMAT_UNORM*/   0, /*DATA_FORMAT_8_8*/          3>;
defm : Gfx10PlusBufferFormat< /*FORMAT_8_8_SNORM*/            0x0F,  8, 2, /*NUM_FORMAT_SNORM*/   1, /*DATA_FORMAT_8_8*/          3>;
defm : Gfx10PlusBufferFormat< /*FORMAT_8_8_USCALED*/          0x10,  8, 2, /*NUM_FORMAT_USCALED*/ 2, /*DATA_FORMAT_8_8*/          3>;
defm : Gfx10PlusBufferFormat< /*FORMAT_8_8_SSCALED*/          0x11,  8, 2, /*NUM_FORMAT_SSCALED*/ 3, /*DATA_FORMAT_8_8*/          3>;
defm : Gfx10PlusBufferFormat< /*FORMAT_8_8_UINT*/             0x12,  8, 2, /*NUM_FORMAT_UINT*/    4, /*DATA_FORMAT_8_8*/          3>;
defm : Gfx10PlusBufferFormat< /*FORMAT_8_8_SINT*/             0x13,  8, 2, /*NUM_FORMAT_SINT*/    5, /*DATA_FORMAT_8_8*/          3>;
defm : Gfx10PlusBufferFormat< /*FORMAT_32_UINT*/              0x14, 32, 1, /*NUM_FORMAT_UINT*/    4, /*DATA_FORMAT_32*/           4>;
defm : Gfx10PlusBufferFormat< /*FORMAT_32_SINT*/              0x15, 32, 1, /*NUM_FORMAT_SINT*/    5, /*DATA_FORMAT_32*/           4>;
defm : Gfx10PlusBufferFormat< /*FORMAT_32_FLOAT*/             0x16, 32, 1, /*NUM_FORMAT_FLOAT*/   7, /*DATA_FORMAT_32*/           4>;
defm : Gfx10PlusBufferFormat< /*FORMAT_16_16_UNORM*/          0x17, 16, 2, /*NUM_FORMAT_UNORM*/   0, /*DATA_FORMAT_16_16*/        5>;
defm : Gfx10PlusBufferFormat< /*FORMAT_16_16_SNORM*/          0x18, 16, 2, /*NUM_FORMAT_SNORM*/   1, /*DATA_FORMAT_16_16*/        5>;
defm : Gfx10PlusBufferFormat< /*FORMAT_16_16_USCALED*/        0x19, 16, 2, /*NUM_FORMAT_USCALED*/ 2, /*DATA_FORMAT_16_16*/        5>;
defm : Gfx10PlusBufferFormat< /*FORMAT_16_16_SSCALED*/        0x1A, 16, 2, /*NUM_FORMAT_SSCALED*/ 3, /*DATA_FORMAT_16_16*/        5>;
defm : Gfx10PlusBufferFormat< /*FORMAT_16_16_UINT*/           0x1B, 16, 2, /*NUM_FORMAT_UINT*/    4, /*DATA_FORMAT_16_16*/        5>;
defm : Gfx10PlusBufferFormat< /*FORMAT_16_16_SINT*/           0x1C, 16, 2, /*NUM_FORMAT_SINT*/    5, /*DATA_FORMAT_16_16*/        5>;
defm : Gfx10PlusBufferFormat< /*FORMAT_16_16_FLOAT*/          0x1D, 16, 2, /*NUM_FORMAT_FLOAT*/   7, /*DATA_FORMAT_16_16*/        5>;

// Buffer formats with equal component sizes (GFX10 only)
def : Gfx10BufferFormat< /*FORMAT_8_8_8_8_UNORM*/        0x38,  8, 4, /*NUM_FORMAT_UNORM*/   0, /*DATA_FORMAT_8_8_8_8*/     10>;
def : Gfx10BufferFormat< /*FORMAT_8_8_8_8_SNORM*/        0x39,  8, 4, /*NUM_FORMAT_SNORM*/   1, /*DATA_FORMAT_8_8_8_8*/     10>;
def : Gfx10BufferFormat< /*FORMAT_8_8_8_8_USCALED*/      0x3A,  8, 4, /*NUM_FORMAT_USCALED*/ 2, /*DATA_FORMAT_8_8_8_8*/     10>;
def : Gfx10BufferFormat< /*FORMAT_8_8_8_8_SSCALED*/      0x3B,  8, 4, /*NUM_FORMAT_SSCALED*/ 3, /*DATA_FORMAT_8_8_8_8*/     10>;
def : Gfx10BufferFormat< /*FORMAT_8_8_8_8_UINT*/         0x3C,  8, 4, /*NUM_FORMAT_UINT*/    4, /*DATA_FORMAT_8_8_8_8*/     10>;
def : Gfx10BufferFormat< /*FORMAT_8_8_8_8_SINT*/         0x3D,  8, 4, /*NUM_FORMAT_SINT*/    5, /*DATA_FORMAT_8_8_8_8*/     10>;
def : Gfx10BufferFormat< /*FORMAT_32_32_UINT*/           0x3E, 32, 2, /*NUM_FORMAT_UINT*/    4, /*DATA_FORMAT_32_32*/       11>;
def : Gfx10BufferFormat< /*FORMAT_32_32_SINT*/           0x3F, 32, 2, /*NUM_FORMAT_SINT*/    5, /*DATA_FORMAT_32_32*/       11>;
def : Gfx10BufferFormat< /*FORMAT_32_32_FLOAT*/          0x40, 32, 2, /*NUM_FORMAT_FLOAT*/   7, /*DATA_FORMAT_32_32*/       11>;
def : Gfx10BufferFormat< /*FORMAT_16_16_16_16_UNORM*/    0x41, 16, 4, /*NUM_FORMAT_UNORM*/   0, /*DATA_FORMAT_16_16_16_16*/ 12>;
def : Gfx10BufferFormat< /*FORMAT_16_16_16_16_SNORM*/    0x42, 16, 4, /*NUM_FORMAT_SNORM*/   1, /*DATA_FORMAT_16_16_16_16*/ 12>;
def : Gfx10BufferFormat< /*FORMAT_16_16_16_16_USCALED*/  0x43, 16, 4, /*NUM_FORMAT_USCALED*/ 2, /*DATA_FORMAT_16_16_16_16*/ 12>;
def : Gfx10BufferFormat< /*FORMAT_16_16_16_16_SSCALED*/  0x44, 16, 4, /*NUM_FORMAT_SSCALED*/ 3, /*DATA_FORMAT_16_16_16_16*/ 12>;
def : Gfx10BufferFormat< /*FORMAT_16_16_16_16_UINT*/     0x45, 16, 4, /*NUM_FORMAT_UINT*/    4, /*DATA_FORMAT_16_16_16_16*/ 12>;
def : Gfx10BufferFormat< /*FORMAT_16_16_16_16_SINT*/     0x46, 16, 4, /*NUM_FORMAT_SINT*/    5, /*DATA_FORMAT_16_16_16_16*/ 12>;
def : Gfx10BufferFormat< /*FORMAT_16_16_16_16_FLOAT*/    0x47, 16, 4, /*NUM_FORMAT_FLOAT*/   7, /*DATA_FORMAT_16_16_16_16*/ 12>;
def : Gfx10BufferFormat< /*FORMAT_32_32_32_UINT*/        0x48, 32, 3, /*NUM_FORMAT_UINT*/    4, /*DATA_FORMAT_32_32_32*/    13>;
def : Gfx10BufferFormat< /*FORMAT_32_32_32_SINT*/        0x49, 32, 3, /*NUM_FORMAT_SINT*/    5, /*DATA_FORMAT_32_32_32*/    13>;
def : Gfx10BufferFormat< /*FORMAT_32_32_32_FLOAT*/       0x4A, 32, 3, /*NUM_FORMAT_FLOAT*/   7, /*DATA_FORMAT_32_32_32*/    13>;
def : Gfx10BufferFormat< /*FORMAT_32_32_32_32_UINT*/     0x4B, 32, 4, /*NUM_FORMAT_UINT*/    4, /*DATA_FORMAT_32_32_32_32*/ 14>;
def : Gfx10BufferFormat< /*FORMAT_32_32_32_32_SINT*/     0x4C, 32, 4, /*NUM_FORMAT_SINT*/    5, /*DATA_FORMAT_32_32_32_32*/ 14>;
def : Gfx10BufferFormat< /*FORMAT_32_32_32_32_FLOAT*/    0x4D, 32, 4, /*NUM_FORMAT_FLOAT*/   7, /*DATA_FORMAT_32_32_32_32*/ 14>;

// Buffer formats with equal component sizes (GFX11 and later)
def : Gfx11PlusBufferFormat< /*FORMAT_8_8_8_8_UNORM*/        0x2A,  8, 4, /*NUM_FORMAT_UNORM*/   0, /*DATA_FORMAT_8_8_8_8*/     10>;
def : Gfx11PlusBufferFormat< /*FORMAT_8_8_8_8_SNORM*/        0x2B,  8, 4, /*NUM_FORMAT_SNORM*/   1, /*DATA_FORMAT_8_8_8_8*/     10>;
def : Gfx11PlusBufferFormat< /*FORMAT_8_8_8_8_USCALED*/      0x2C,  8, 4, /*NUM_FORMAT_USCALED*/ 2, /*DATA_FORMAT_8_8_8_8*/     10>;
def : Gfx11PlusBufferFormat< /*FORMAT_8_8_8_8_SSCALED*/      0x2D,  8, 4, /*NUM_FORMAT_SSCALED*/ 3, /*DATA_FORMAT_8_8_8_8*/     10>;
def : Gfx11PlusBufferFormat< /*FORMAT_8_8_8_8_UINT*/         0x2E,  8, 4, /*NUM_FORMAT_UINT*/    4, /*DATA_FORMAT_8_8_8_8*/     10>;
def : Gfx11PlusBufferFormat< /*FORMAT_8_8_8_8_SINT*/         0x2F,  8, 4, /*NUM_FORMAT_SINT*/    5, /*DATA_FORMAT_8_8_8_8*/     10>;
def : Gfx11PlusBufferFormat< /*FORMAT_32_32_UINT*/           0x30, 32, 2, /*NUM_FORMAT_UINT*/    4, /*DATA_FORMAT_32_32*/       11>;
def : Gfx11PlusBufferFormat< /*FORMAT_32_32_SINT*/           0x31, 32, 2, /*NUM_FORMAT_SINT*/    5, /*DATA_FORMAT_32_32*/       11>;
def : Gfx11PlusBufferFormat< /*FORMAT_32_32_FLOAT*/          0x32, 32, 2, /*NUM_FORMAT_FLOAT*/   7, /*DATA_FORMAT_32_32*/       11>;
def : Gfx11PlusBufferFormat< /*FORMAT_16_16_16_16_UNORM*/    0x33, 16, 4, /*NUM_FORMAT_UNORM*/   0, /*DATA_FORMAT_16_16_16_16*/ 12>;
def : Gfx11PlusBufferFormat< /*FORMAT_16_16_16_16_SNORM*/    0x34, 16, 4, /*NUM_FORMAT_SNORM*/   1, /*DATA_FORMAT_16_16_16_16*/ 12>;
def : Gfx11PlusBufferFormat< /*FORMAT_16_16_16_16_USCALED*/  0x35, 16, 4, /*NUM_FORMAT_USCALED*/ 2, /*DATA_FORMAT_16_16_16_16*/ 12>;
def : Gfx11PlusBufferFormat< /*FORMAT_16_16_16_16_SSCALED*/  0x36, 16, 4, /*NUM_FORMAT_SSCALED*/ 3, /*DATA_FORMAT_16_16_16_16*/ 12>;
def : Gfx11PlusBufferFormat< /*FORMAT_16_16_16_16_UINT*/     0x37, 16, 4, /*NUM_FORMAT_UINT*/    4, /*DATA_FORMAT_16_16_16_16*/ 12>;
def : Gfx11PlusBufferFormat< /*FORMAT_16_16_16_16_SINT*/     0x38, 16, 4, /*NUM_FORMAT_SINT*/    5, /*DATA_FORMAT_16_16_16_16*/ 12>;
def : Gfx11PlusBufferFormat< /*FORMAT_16_16_16_16_FLOAT*/    0x39, 16, 4, /*NUM_FORMAT_FLOAT*/   7, /*DATA_FORMAT_16_16_16_16*/ 12>;
def : Gfx11PlusBufferFormat< /*FORMAT_32_32_32_UINT*/        0x3A, 32, 3, /*NUM_FORMAT_UINT*/    4, /*DATA_FORMAT_32_32_32*/    13>;
def : Gfx11PlusBufferFormat< /*FORMAT_32_32_32_SINT*/        0x3B, 32, 3, /*NUM_FORMAT_SINT*/    5, /*DATA_FORMAT_32_32_32*/    13>;
def : Gfx11PlusBufferFormat< /*FORMAT_32_32_32_FLOAT*/       0x3C, 32, 3, /*NUM_FORMAT_FLOAT*/   7, /*DATA_FORMAT_32_32_32*/    13>;
def : Gfx11PlusBufferFormat< /*FORMAT_32_32_32_32_UINT*/     0x3D, 32, 4, /*NUM_FORMAT_UINT*/    4, /*DATA_FORMAT_32_32_32_32*/ 14>;
def : Gfx11PlusBufferFormat< /*FORMAT_32_32_32_32_SINT*/     0x3E, 32, 4, /*NUM_FORMAT_SINT*/    5, /*DATA_FORMAT_32_32_32_32*/ 14>;
def : Gfx11PlusBufferFormat< /*FORMAT_32_32_32_32_FLOAT*/    0x3F, 32, 4, /*NUM_FORMAT_FLOAT*/   7, /*DATA_FORMAT_32_32_32_32*/ 14>;

class SourceOfDivergence<Intrinsic intr> {
  Intrinsic Intr = intr;
}

def SourcesOfDivergence : GenericTable {
  let FilterClass = "SourceOfDivergence";
  let Fields = ["Intr"];

  let PrimaryKey = ["Intr"];
  let PrimaryKeyName = "lookupSourceOfDivergence";
}

def : SourceOfDivergence<int_amdgcn_workitem_id_x>;
def : SourceOfDivergence<int_amdgcn_workitem_id_y>;
def : SourceOfDivergence<int_amdgcn_workitem_id_z>;
def : SourceOfDivergence<int_amdgcn_interp_mov>;
def : SourceOfDivergence<int_amdgcn_interp_p1>;
def : SourceOfDivergence<int_amdgcn_interp_p2>;
def : SourceOfDivergence<int_amdgcn_interp_p1_f16>;
def : SourceOfDivergence<int_amdgcn_interp_p2_f16>;
def : SourceOfDivergence<int_amdgcn_lds_direct_load>;
def : SourceOfDivergence<int_amdgcn_lds_param_load>;
def : SourceOfDivergence<int_amdgcn_mbcnt_hi>;
def : SourceOfDivergence<int_amdgcn_mbcnt_lo>;
def : SourceOfDivergence<int_r600_read_tidig_x>;
def : SourceOfDivergence<int_r600_read_tidig_y>;
def : SourceOfDivergence<int_r600_read_tidig_z>;
def : SourceOfDivergence<int_amdgcn_global_atomic_fmin_num>;
def : SourceOfDivergence<int_amdgcn_global_atomic_fmax_num>;
def : SourceOfDivergence<int_amdgcn_global_atomic_ordered_add_b64>;
def : SourceOfDivergence<int_amdgcn_flat_atomic_fmin_num>;
def : SourceOfDivergence<int_amdgcn_flat_atomic_fmax_num>;
// The results of buffer atomics differ in each lane.
foreach intr = AMDGPUBufferIntrinsics in {
  if !or(!eq(intr.BufferKind, AMDGPUBufferKindAtomicRMW), !eq(intr.BufferKind, AMDGPUBufferKindAtomicCmpSwap)) then
    def : SourceOfDivergence<!cast<Intrinsic>(intr)>;
}
def : SourceOfDivergence<int_amdgcn_ps_live>;
def : SourceOfDivergence<int_amdgcn_live_mask>;
def : SourceOfDivergence<int_amdgcn_ds_swizzle>;
def : SourceOfDivergence<int_amdgcn_ds_ordered_add>;
def : SourceOfDivergence<int_amdgcn_ds_ordered_swap>;
def : SourceOfDivergence<int_amdgcn_ds_add_gs_reg_rtn>;
def : SourceOfDivergence<int_amdgcn_ds_sub_gs_reg_rtn>;
def : SourceOfDivergence<int_amdgcn_ds_atomic_barrier_arrive_rtn_b64>;
def : SourceOfDivergence<int_amdgcn_permlane16>;
def : SourceOfDivergence<int_amdgcn_permlanex16>;
def : SourceOfDivergence<int_amdgcn_permlane16_var>;
def : SourceOfDivergence<int_amdgcn_permlanex16_var>;
def : SourceOfDivergence<int_amdgcn_permlane64>;
def : SourceOfDivergence<int_amdgcn_permlane_bcast>;
def : SourceOfDivergence<int_amdgcn_permlane_up>;
def : SourceOfDivergence<int_amdgcn_permlane_down>;
def : SourceOfDivergence<int_amdgcn_permlane_xor>;
def : SourceOfDivergence<int_amdgcn_permlane_idx_gen>;
def : SourceOfDivergence<int_amdgcn_exclusive_scan_sum_i32>;
def : SourceOfDivergence<int_amdgcn_exclusive_scan_sum_u32>;
def : SourceOfDivergence<int_amdgcn_exclusive_scan_xor_b32>;
def : SourceOfDivergence<int_amdgcn_exclusive_scan_or_b32>;
def : SourceOfDivergence<int_amdgcn_exclusive_scan_and_b32>;
def : SourceOfDivergence<int_amdgcn_exclusive_scan_min_i16>;
def : SourceOfDivergence<int_amdgcn_exclusive_scan_min_u16>;
def : SourceOfDivergence<int_amdgcn_exclusive_scan_min_i32>;
def : SourceOfDivergence<int_amdgcn_exclusive_scan_min_u32>;
def : SourceOfDivergence<int_amdgcn_exclusive_scan_max_i16>;
def : SourceOfDivergence<int_amdgcn_exclusive_scan_max_u16>;
def : SourceOfDivergence<int_amdgcn_exclusive_scan_max_i32>;
def : SourceOfDivergence<int_amdgcn_exclusive_scan_max_u32>;
def : SourceOfDivergence<int_amdgcn_mov_dpp>;
def : SourceOfDivergence<int_amdgcn_mov_dpp8>;
def : SourceOfDivergence<int_amdgcn_update_dpp>;
def : SourceOfDivergence<int_amdgcn_writelane>;
def : SourceOfDivergence<int_amdgcn_init_whole_wave>;
def : SourceOfDivergence<int_amdgcn_permlane16_swap>;
def : SourceOfDivergence<int_amdgcn_permlane32_swap>;
def : SourceOfDivergence<int_amdgcn_set_inactive>;
def : SourceOfDivergence<int_amdgcn_set_inactive_chain_arg>;

foreach intr = AMDGPUMFMAIntrinsics908 in
def : SourceOfDivergence<intr>;
foreach intr = AMDGPUMFMAIntrinsics90A in
def : SourceOfDivergence<intr>;
foreach intr = AMDGPUMFMAIntrinsics940 in
def : SourceOfDivergence<intr>;
foreach intr = AMDGPUMFMAIntrinsics950 in
def : SourceOfDivergence<intr>;
foreach intr = AMDGPUWMMAIntrinsicsGFX11 in
def : SourceOfDivergence<intr>;
foreach intr = AMDGPUWMMAIntrinsicsGFX12 in
def : SourceOfDivergence<intr>;
foreach intr = AMDGPUWMMAIntrinsicsGFX1250 in
def : SourceOfDivergence<intr>;
foreach intr = AMDGPUSWMMACIntrinsicsGFX1250 in
def : SourceOfDivergence<intr>;

def : SourceOfDivergence<int_amdgcn_global_load_tr_b64>;
def : SourceOfDivergence<int_amdgcn_global_load_tr_b128>;
def : SourceOfDivergence<int_amdgcn_global_load_tr4_b64>;
def : SourceOfDivergence<int_amdgcn_global_load_tr6_b96>;
def : SourceOfDivergence<int_amdgcn_ds_load_tr8_b64>;
def : SourceOfDivergence<int_amdgcn_ds_load_tr16_b128>;
def : SourceOfDivergence<int_amdgcn_ds_load_tr4_b64>;
def : SourceOfDivergence<int_amdgcn_ds_load_tr6_b96>;

def : SourceOfDivergence<int_amdgcn_ds_read_tr4_b64>;
def : SourceOfDivergence<int_amdgcn_ds_read_tr6_b96>;
def : SourceOfDivergence<int_amdgcn_ds_read_tr8_b64>;
def : SourceOfDivergence<int_amdgcn_ds_read_tr16_b64>;

// The dummy boolean output is divergent from the IR's perspective,
// but the mask results are uniform. These produce a divergent and
// uniform result, so the returned struct is collectively divergent.
// isAlwaysUniform can override the extract of the uniform component.
def : SourceOfDivergence<int_amdgcn_if>;
def : SourceOfDivergence<int_amdgcn_else>;
def : SourceOfDivergence<int_amdgcn_loop>;
def : SourceOfDivergence<int_amdgcn_inverse_ballot>;

foreach intr = AMDGPUImageDimAtomicIntrinsics in
def : SourceOfDivergence<intr>;

def : SourceOfDivergence<int_amdgcn_dead>;
def : SourceOfDivergence<int_amdgcn_call_whole_wave>;

class AlwaysUniform<Intrinsic intr> {
  Intrinsic Intr = intr;
}

def UniformIntrinsics : GenericTable {
  let FilterClass = "AlwaysUniform";
  let Fields = ["Intr"];

  let PrimaryKey = ["Intr"];
  let PrimaryKeyName = "lookupAlwaysUniform";
}

def : AlwaysUniform<int_amdgcn_readfirstlane>;
def : AlwaysUniform<int_amdgcn_readlane>;
def : AlwaysUniform<int_amdgcn_ballot>;
def : AlwaysUniform<int_amdgcn_if_break>;
def : AlwaysUniform<int_amdgcn_cluster_workgroup_id_x>;
def : AlwaysUniform<int_amdgcn_cluster_workgroup_id_y>;
def : AlwaysUniform<int_amdgcn_cluster_workgroup_id_z>;
def : AlwaysUniform<int_amdgcn_cluster_workgroup_flat_id>;
def : AlwaysUniform<int_amdgcn_cluster_workgroup_max_id_x>;
def : AlwaysUniform<int_amdgcn_cluster_workgroup_max_id_y>;
def : AlwaysUniform<int_amdgcn_cluster_workgroup_max_id_z>;
def : AlwaysUniform<int_amdgcn_cluster_workgroup_max_flat_id>;
def : AlwaysUniform<int_amdgcn_workgroup_id_x>;
def : AlwaysUniform<int_amdgcn_workgroup_id_y>;
def : AlwaysUniform<int_amdgcn_workgroup_id_z>;
def : AlwaysUniform<int_amdgcn_s_alloc_vgpr>;
def : AlwaysUniform<int_amdgcn_s_get_barrier_state>;
def : AlwaysUniform<int_amdgcn_s_get_named_barrier_state>;
def : AlwaysUniform<int_amdgcn_s_getpc>;
def : AlwaysUniform<int_amdgcn_s_getreg>;
def : AlwaysUniform<int_amdgcn_s_memrealtime>;
def : AlwaysUniform<int_amdgcn_s_memtime>;
def : AlwaysUniform<int_amdgcn_s_quadmask>;
def : AlwaysUniform<int_amdgcn_s_wqm>;

def AMDGPUImageDMaskIntrinsicTable : GenericTable {
  let FilterClass = "AMDGPUImageDMaskIntrinsic";
  let Fields = ["Intr"];

  let PrimaryKey = ["Intr"];
  let PrimaryKeyName = "getAMDGPUImageDMaskIntrinsic";
  let PrimaryKeyEarlyOut = 1;
}

def LDSDMAIntrinsicTable : GenericTable {
  let FilterClass = "AMDGPULDSDMAIntrinsic";
  let CppTypeName = "LDSDMAIntrinsic";
  let Fields = ["Intr"];

  let PrimaryKey = ["Intr"];
  let PrimaryKeyName = "lookupLDSDMAIntrinsic";
}

def VLdStIdxOpcodeInfoTable : GenericTable {
  let FilterClass = "VLdStIdxOpcodeInfo";
  let CppTypeName = "VLdStIdxOpcodeInfo";
  let Fields = ["Opcode", "BitWidth", "IsStore", "IsGPRIdx"];
  let PrimaryKey = ["Opcode"];
  let PrimaryKeyName = "getVLdStIdxOpcodeInfoByOpcode";
}

def getVLdStIdxOpcodeInfoByKeyImpl : SearchIndex {
  let Table = VLdStIdxOpcodeInfoTable;
  let Key = ["BitWidth", "IsStore", "IsGPRIdx"];
}

class ExponentOnlyScaleIntrinsic<AMDGPUExponentOnlyScale intr> {
  Intrinsic Intr = !cast<Intrinsic>(intr);
  bits<8> ScaleArg = intr.ScaleArg;
  assert !eq(Intr.ParamTypes[intr.ScaleArg], llvm_float_ty),
    "scale argument of " # !cast<string>(intr) # " must be f32";
}

def ExponentOnlyScaleIntrinsics : GenericTable {
  let FilterClass = "ExponentOnlyScaleIntrinsic";
  let Fields = ["Intr", "ScaleArg"];

  let PrimaryKey = ["Intr"];
  let PrimaryKeyName = "lookupExponentOnlyScaleIntrinsic";
  let PrimaryKeyEarlyOut = 1;
}

foreach intr = !instances<AMDGPUExponentOnlyScale>() in
  def : ExponentOnlyScaleIntrinsic<intr>;

//===----------------------------------------------------------------------===//
// GFX1250 Blocking Cycles Tables
//===----------------------------------------------------------------------===//
//
// BlockingCycles indicates how many cycles until a new instruction can acquire
// a resource, allowing modeling of multi-cycle blocking.
//
//===----------------------------------------------------------------------===//

class GFX1250BlockingCyclesEntry<Instruction inst, bits<8> cycles> {
  Instruction Inst = inst;
  bits<8> GFX1250BlockingCycles = cycles;
}

def GFX1250BlockingCyclesTable : GenericTable {
  let FilterClass = "GFX1250BlockingCyclesEntry";
  let CppTypeName = "AMDGPUBlockingCyclesInfo";
  let Fields = ["Inst", "GFX1250BlockingCycles"];
  let PrimaryKey = ["Inst"];
  let PrimaryKeyName = "getGFX1250BlockingCyclesInfo";
}

// --- DPP instructions ---
// All VOP DPP pseudos get blocking cycles. Default is 2, with overrides for
// F64/I64 operations (32 cycles) and integer multiply/MAD (4 cycles).
// FMAC_F64 and *_CLAMP_F64 are excluded from the 32-cycle override.
// Only specific MAD/MUL variants (not all with MAD_/MUL_HI/MUL_LO substrings)
// get 4 cycles.
foreach inst = !instances<VOP_DPP_Pseudo>(".*") in {
  defvar name = !cast<string>(inst);
  defvar is32 = !and(
    !or(!ne(!find(name, "F64"), -1),
        !ne(!find(name, "MUL_U64"), -1),
        !ne(!find(name, "MIN_I64"), -1),
        !ne(!find(name, "MAX_I64"), -1),
        !ne(!find(name, "MIN_U64"), -1),
        !ne(!find(name, "MAX_U64"), -1)),
    !eq(!find(name, "FMAC"), -1),
    !eq(!find(name, "CLAMP"), -1));
  defvar is4 = !or(
    !and(!ne(!find(name, "MAD_U32"), -1),
         !eq(!find(name, "MAD_U32_U"), -1)),
    !ne(!find(name, "MAD_I64"), -1),
    !ne(!find(name, "MAD_U64"), -1),
    !ne(!find(name, "MAD_NC_"), -1),
    !and(!ne(!find(name, "MUL_HI_I32"), -1),
         !eq(!find(name, "MUL_HI_I32_I"), -1)),
    !and(!ne(!find(name, "MUL_HI_U32"), -1),
         !eq(!find(name, "MUL_HI_U32_U"), -1)),
    !ne(!find(name, "MUL_LO_U32"), -1));
  def : GFX1250BlockingCyclesEntry<inst, !if(is32, 32, !if(is4, 4, 2))>;
}

// --- DPP8 instructions ---
// All VOP DPP8 pseudos get blocking cycles. Default is 2, with override for
// F64 operations (32 cycles). *_CLAMP_F64 variants are excluded.
foreach inst = !instances<VOP_DPP8_Pseudo>(".*") in {
  defvar name = !cast<string>(inst);
  defvar is32 = !and(!ne(!find(name, "F64"), -1),
                     !eq(!find(name, "CLAMP"), -1));
  def : GFX1250BlockingCyclesEntry<inst, !if(is32, 32, 2)>;
}

// --- Non-DPP/DPP8 instructions ---

// BlockingCycles = 32: F64 transcendentals and rounding/specials
foreach inst = !instances<Instruction>("^V_(RCP|RSQ|SQRT|FRACT|TRUNC|CEIL|RNDNE|FLOOR|FREXP_EXP_I32|FREXP_MANT)_F64_e(32|64)$") in
  def : GFX1250BlockingCyclesEntry<inst, 32>;

// BlockingCycles = 32: F64 conversions
foreach inst = !instances<Instruction>("^V_CVT_(I32_F64|F64_I32|F32_F64|F64_F32|U32_F64|F64_U32)_e(32|64)$") in
  def : GFX1250BlockingCyclesEntry<inst, 32>;

// BlockingCycles = 32: F64 arithmetic (VOP3)
foreach inst = !instances<Instruction>("^V_(ADD|MUL|FMA|LDEXP|DIV_FIXUP|DIV_FMAS|DIV_SCALE|MIN_NUM|MAX_NUM|MINIMUM|MAXIMUM|MIN|MAX)_F64_e64$") in
  def : GFX1250BlockingCyclesEntry<inst, 32>;
foreach inst = !instances<Instruction>("^V_(MIN|MAX)_NUM_F64_e32$") in
  def : GFX1250BlockingCyclesEntry<inst, 32>;
foreach inst = !instances<Instruction>("^V_TRIG_PREOP_F64_e64$") in
  def : GFX1250BlockingCyclesEntry<inst, 32>;

// BlockingCycles = 32: F64 pseudo arithmetic
foreach inst = !instances<Instruction>("^V_(ADD|MUL)_F64_pseudo_e(32|64)$") in
  def : GFX1250BlockingCyclesEntry<inst, 32>;

// BlockingCycles = 32: F64 comparisons (CMP, CMPX, CMPX_nosdst, CLASS)
foreach inst = !instances<Instruction>("^V_CMPX?_.*_F64(_nosdst)?_e(32|64)$") in
  def : GFX1250BlockingCyclesEntry<inst, 32>;

// BlockingCycles = 32: 64-bit integer min/max
foreach inst = !instances<Instruction>("^V_(MIN|MAX)_(I64|U64)_e64$") in
  def : GFX1250BlockingCyclesEntry<inst, 32>;

// BlockingCycles = 32: V_MUL_U64
foreach inst = !instances<Instruction>("^V_MUL_U64_e(32|64)$") in
  def : GFX1250BlockingCyclesEntry<inst, 32>;

// BlockingCycles = 16: CVT_SCALEF32_PK32 instructions
foreach inst = !instances<Instruction>("^V_CVT_SCALEF32_SR_PK32_.*_e64$") in
  def : GFX1250BlockingCyclesEntry<inst, 16>;
foreach inst = !instances<Instruction>("^V_CVT_SCALEF32_PK32_((BF16|F16|F32)_(BF6|FP6)|(BF6|FP6)_(BF16|F16))_e64$") in
  def : GFX1250BlockingCyclesEntry<inst, 16>;

// BlockingCycles = 8: CVT_SCALE(F32)_PK16 instructions
foreach inst = !instances<Instruction>("^V_CVT_SCALE(F32)?_(2X|SR_)?PK16_.*_e64$") in
  def : GFX1250BlockingCyclesEntry<inst, 8>;

// BlockingCycles = 4: Integer multiply
foreach inst = !instances<Instruction>("^V_(MUL_LO_U32|MUL_HI_U32|MUL_HI_I32|MAD_U32)_e64$") in
  def : GFX1250BlockingCyclesEntry<inst, 4>;
foreach inst = !instances<Instruction>("^V_MAD_(NC_)?(U64_U32|I64_I32)_e64$") in
  def : GFX1250BlockingCyclesEntry<inst, 4>;

// BlockingCycles = 4: QSAD/MQSAD
foreach inst = !instances<Instruction>("^V_M?QSAD_.*_e64$") in
  def : GFX1250BlockingCyclesEntry<inst, 4>;

// BlockingCycles = 4: DOT instructions
foreach inst = !instances<Instruction>("^V_DOT(4_I32_IU8|8_I32_IU4)$") in
  def : GFX1250BlockingCyclesEntry<inst, 4>;

// BlockingCycles = 4: CVT_SCALE(F32)_PK8 and PERM_PK16 instructions
foreach inst = !instances<Instruction>("^V_CVT_SCALE(F32)?_(SR_)?PK8_.*_e64$") in
  def : GFX1250BlockingCyclesEntry<inst, 4>;
foreach inst = !instances<Instruction>("^V_PERM_PK16_.*_e64$") in
  def : GFX1250BlockingCyclesEntry<inst, 4>;

// BlockingCycles = 2: F32/F16/BF16 transcendentals
foreach inst = !instances<Instruction>("^V_(EXP|LOG|RCP|RSQ|SQRT|SIN|COS|TANH)_F32_(e(32|64)|sdwa)$") in
  def : GFX1250BlockingCyclesEntry<inst, 2>;
foreach inst = !instances<Instruction>("^V_RCP_IFLAG_F32_(e(32|64)|sdwa)$") in
  def : GFX1250BlockingCyclesEntry<inst, 2>;
foreach inst = !instances<Instruction>("^V_(EXP|LOG|RCP|RSQ|SQRT|SIN|COS|TANH)_F16_(e(32|64)|sdwa|(fake16|t16)_(e(32|64)|sdwa))$") in
  def : GFX1250BlockingCyclesEntry<inst, 2>;
foreach inst = !instances<Instruction>("^V_(EXP|LOG|RCP|RSQ|SQRT|SIN|COS|TANH)_BF16_(e(32|64)|sdwa|(fake16|t16)_(e(32|64)|sdwa))$") in
  def : GFX1250BlockingCyclesEntry<inst, 2>;

// BlockingCycles = 2: Pseudo scalar transcendentals
foreach inst = !instances<Instruction>("^V_S_(EXP|LOG|RCP|RSQ|SQRT)_(F32|F16(_noinline)?)_e64$") in
  def : GFX1250BlockingCyclesEntry<inst, 2>;

// BlockingCycles = 2: 64-bit shifts/moves
foreach inst = !instances<Instruction>("^V_(LSHLREV_B64|LSHRREV_B64|ASHRREV_I64)_e64$") in
  def : GFX1250BlockingCyclesEntry<inst, 2>;
foreach inst = !instances<Instruction>("^V_MOV_B64_e(32|64)$") in
  def : GFX1250BlockingCyclesEntry<inst, 2>;
foreach inst = !instances<Instruction>("^V_LSHL_ADD_U64_e64$") in
  def : GFX1250BlockingCyclesEntry<inst, 2>;
foreach inst = !instances<Instruction>("^V_LSHLREV_B64_pseudo_e(32|64)$") in
  def : GFX1250BlockingCyclesEntry<inst, 2>;

// BlockingCycles = 2: 64-bit integer comparisons
foreach inst = !instances<Instruction>("^V_CMPX?_.*_(I64|U64)(_nosdst)?_e(32|64)$") in
  def : GFX1250BlockingCyclesEntry<inst, 2>;

// BlockingCycles = 2: ADD/SUB U64
foreach inst = !instances<Instruction>("^V_(ADD|SUB)_U64_e(32|64)$") in
  def : GFX1250BlockingCyclesEntry<inst, 2>;

// BlockingCycles = 2: SWAP and PERMLANE_SWAP instructions
foreach inst = !instances<Instruction>("^V_SWAP_B(16|32)$") in
  def : GFX1250BlockingCyclesEntry<inst, 2>;
foreach inst = !instances<Instruction>("^V_SWAPREL_B32$") in
  def : GFX1250BlockingCyclesEntry<inst, 2>;
foreach inst = !instances<Instruction>("^V_PERMLANE(16|32)_SWAP_B32_e(32|64)$") in
  def : GFX1250BlockingCyclesEntry<inst, 2>;
