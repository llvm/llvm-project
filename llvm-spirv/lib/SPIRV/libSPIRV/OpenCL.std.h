/*
** Copyright (c) 2015 The Khronos Group Inc.
**
** Permission is hereby granted, free of charge, to any person obtaining a copy
** of this software and/or associated documentation files (the "Materials"),
** to deal in the Materials without restriction, including without limitation
** the rights to use, copy, modify, merge, publish, distribute, sublicense,
** and/or sell copies of the Materials, and to permit persons to whom the
** Materials are furnished to do so, subject to the following conditions:
**
** The above copyright notice and this permission notice shall be included in
** all copies or substantial portions of the Materials.
**
** MODIFICATIONS TO THIS FILE MAY MEAN IT NO LONGER ACCURATELY REFLECTS KHRONOS
** STANDARDS. THE UNMODIFIED, NORMATIVE VERSIONS OF KHRONOS SPECIFICATIONS AND
** HEADER INFORMATION ARE LOCATED AT https://www.khronos.org/registry/ 
**
** THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
** OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
** FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
** THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
** LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
** FROM,OUT OF OR IN CONNECTION WITH THE MATERIALS OR THE USE OR OTHER DEALINGS
** IN THE MATERIALS.
*/

//
// Author: Boaz Ouriel, Intel
//

namespace OpenCLLIB {

enum Entrypoints {

    // math functions
    Acos = 0,
    Acosh = 1,
    Acospi = 2,
    Asin = 3,
    Asinh = 4,
    Asinpi = 5,
    Atan = 6,
    Atan2 = 7,
    Atanh = 8,
    Atanpi = 9,
    Atan2pi = 10,
    Cbrt = 11,
    Ceil = 12,
    Copysign = 13,
    Cos = 14,
    Cosh = 15,
    Cospi = 16,
    Erfc = 17,
    Erf = 18,
    Exp = 19,
    Exp2 = 20,
    Exp10 = 21,
    Expm1 = 22,
    Fabs = 23,
    Fdim = 24,
    Floor = 25,
    Fma = 26,
    Fmax = 27,
    Fmin = 28,
    Fmod = 29,
    Fract = 30, 
    Frexp = 31,
    Hypot = 32,
    Ilogb = 33,
    Ldexp = 34,
    Lgamma = 35,
    Lgamma_r = 36,
    Log = 37,
    Log2 = 38,
    Log10 = 39,
    Log1p = 40,
    Logb = 41,
    Mad = 42,
    Maxmag = 43,
    Minmag = 44,
    Modf = 45,
    Nan = 46,
    Nextafter = 47,
    Pow = 48,
    Pown = 49,
    Powr = 50,
    Remainder = 51,
    Remquo = 52,
    Rint = 53,
    Rootn = 54,
    Round = 55,
    Rsqrt = 56,
    Sin = 57,
    Sincos = 58,
    Sinh = 59,
    Sinpi = 60,
    Sqrt = 61,
    Tan = 62,
    Tanh = 63,
    Tanpi = 64,
    Tgamma = 65,
    Trunc = 66,
    Half_cos = 67,
    Half_divide = 68,
    Half_exp = 69,
    Half_exp2 = 70,
    Half_exp10 = 71,
    Half_log = 72,
    Half_log2 = 73,
    Half_log10 = 74,
    Half_powr = 75,
    Half_recip = 76,
    Half_rsqrt = 77,
    Half_sin = 78,
    Half_sqrt = 79,
    Half_tan = 80,
    Native_cos = 81,
    Native_divide = 82,
    Native_exp = 83,
    Native_exp2 = 84,
    Native_exp10 = 85,
    Native_log = 86,
    Native_log2 = 87,
    Native_log10 = 88,
    Native_powr = 89,
    Native_recip = 90,
    Native_rsqrt = 91,
    Native_sin = 92,
    Native_sqrt = 93,
    Native_tan = 94,
    
    // Common
    FClamp = 95,
    Degrees = 96,
    FMax_common = 97,
    FMin_common = 98, 
    Mix = 99,
    Radians = 100,
    Step = 101,
    Smoothstep = 102,
    Sign = 103,
    
    // Geometrics
    Cross = 104,
    Distance = 105, 
    Length = 106,
    Normalize = 107,
    Fast_distance = 108,
    Fast_length = 109,
    Fast_normalize = 110,

    // Images - Deprecated 
    Read_imagef = 111,
    Read_imagei = 112,
    Read_imageui = 113,
    Read_imageh = 114,

    Read_imagef_samplerless = 115,
    Read_imagei_samplerless = 116,
    Read_imageui_samplerless = 117,
    Read_imageh_samplerless = 118,

    Write_imagef = 119,
    Write_imagei = 120,
    Write_imageui = 121,
    Write_imageh = 122,
    Read_imagef_mipmap_lod = 123,
    Read_imagei_mipmap_lod = 124,
    Read_imageui_mipmap_lod = 125,
    Read_imagef_mipmap_grad = 126,
    Read_imagei_mipmap_grad = 127,
    Read_imageui_mipmap_grad = 128,

    // Image write with LOD
    Write_imagef_mipmap_lod = 129,
    Write_imagei_mipmap_lod = 130,
    Write_imageui_mipmap_lod = 131,

    // Images - Deprecated
    Get_image_width = 132,
    Get_image_height = 133,
    Get_image_depth = 134,
    Get_image_channel_data_type = 135,
    Get_image_channel_order = 136,
    Get_image_dim = 137,
    Get_image_array_size = 138,
    Get_image_num_samples = 139,
    Get_image_num_mip_levels = 140,
    
    // Integers
    SAbs = 141,
    SAbs_diff = 142,
    SAdd_sat = 143,
    UAdd_sat = 144,
    SHadd = 145,
    UHadd = 146,
    SRhadd = 147,
    URhadd = 148,
    SClamp = 149,
    UClamp = 150, 
    Clz = 151,
    Ctz = 152,    
    SMad_hi = 153,
    UMad_sat = 154,
    SMad_sat = 155,
    SMax = 156,
    UMax = 157,
    SMin = 158,
    UMin = 159,
    SMul_hi = 160,
    Rotate = 161,
    SSub_sat = 162,
    USub_sat = 163,
    U_Upsample = 164,
    S_Upsample = 165,
    Popcount = 166,
    SMad24 = 167,
    UMad24 = 168,
    SMul24 = 169,
    UMul24 = 170,

    // Vector Loads/Stores
    Vloadn = 171,
    Vstoren = 172,
    Vload_half = 173,
    Vload_halfn = 174,
    Vstore_half = 175,
    Vstore_half_r = 176,
    Vstore_halfn = 177,
    Vstore_halfn_r = 178,
    Vloada_halfn = 179,
    Vstorea_halfn = 180,
    Vstorea_halfn_r = 181,

    // Vector Misc 
    Shuffle = 182,
    Shuffle2 = 183,

    // 
    Printf = 184,
    Prefetch = 185,
    
    // Relationals
    Bitselect = 186,
    Select = 187,
        
    // pipes
    Read_pipe = 188,
    Write_pipe = 189,
    Reserve_read_pipe = 190,
    Reserve_write_pipe = 191,
    Commit_read_pipe = 192,
    Commit_write_pipe = 193,
    Is_valid_reserve_id = 194,
    Work_group_reserve_read_pipe = 195,
    Work_group_reserve_write_pipe = 196,
    Work_group_commit_read_pipe = 197,
    Work_group_commit_write_pipe = 198,
    Get_pipe_num_packets = 199,
    Get_pipe_max_packets = 200,
    
    // more integers
    UAbs = 201,
    UAbs_diff = 202,
    UMul_hi = 203,
    UMad_hi = 204,
};

} // end namespace OpenCL20
