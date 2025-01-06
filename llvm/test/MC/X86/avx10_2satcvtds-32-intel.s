// RUN: llvm-mc -triple i386 -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

// CHECK: vcvttsd2sis ecx, xmm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x6d,0xca]
          vcvttsd2sis ecx, xmm2

// CHECK: vcvttsd2sis ecx, xmm2, {sae}
// CHECK: encoding: [0x62,0xf5,0x7f,0x18,0x6d,0xca]
          vcvttsd2sis ecx, xmm2, {sae}

// CHECK: vcvttsd2sis ecx, qword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x6d,0x8c,0xf4,0x00,0x00,0x00,0x10]
          vcvttsd2sis ecx, qword ptr [esp + 8*esi + 268435456]

// CHECK: vcvttsd2sis ecx, qword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x6d,0x8c,0x87,0x23,0x01,0x00,0x00]
          vcvttsd2sis ecx, qword ptr [edi + 4*eax + 291]

// CHECK: vcvttsd2sis ecx, qword ptr [eax]
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x6d,0x08]
          vcvttsd2sis ecx, qword ptr [eax]

// CHECK: vcvttsd2sis ecx, qword ptr [2*ebp - 256]
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x6d,0x0c,0x6d,0x00,0xff,0xff,0xff]
          vcvttsd2sis ecx, qword ptr [2*ebp - 256]

// CHECK: vcvttsd2sis ecx, qword ptr [ecx + 1016]
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x6d,0x49,0x7f]
          vcvttsd2sis ecx, qword ptr [ecx + 1016]

// CHECK: vcvttsd2sis ecx, qword ptr [edx - 1024]
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x6d,0x4a,0x80]
          vcvttsd2sis ecx, qword ptr [edx - 1024]

// CHECK: vcvttsd2usis ecx, xmm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x6c,0xca]
          vcvttsd2usis ecx, xmm2

// CHECK: vcvttsd2usis ecx, xmm2, {sae}
// CHECK: encoding: [0x62,0xf5,0x7f,0x18,0x6c,0xca]
          vcvttsd2usis ecx, xmm2, {sae}

// CHECK: vcvttsd2usis ecx, qword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x6c,0x8c,0xf4,0x00,0x00,0x00,0x10]
          vcvttsd2usis ecx, qword ptr [esp + 8*esi + 268435456]

// CHECK: vcvttsd2usis ecx, qword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x6c,0x8c,0x87,0x23,0x01,0x00,0x00]
          vcvttsd2usis ecx, qword ptr [edi + 4*eax + 291]

// CHECK: vcvttsd2usis ecx, qword ptr [eax]
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x6c,0x08]
          vcvttsd2usis ecx, qword ptr [eax]

// CHECK: vcvttsd2usis ecx, qword ptr [2*ebp - 256]
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x6c,0x0c,0x6d,0x00,0xff,0xff,0xff]
          vcvttsd2usis ecx, qword ptr [2*ebp - 256]

// CHECK: vcvttsd2usis ecx, qword ptr [ecx + 1016]
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x6c,0x49,0x7f]
          vcvttsd2usis ecx, qword ptr [ecx + 1016]

// CHECK: vcvttsd2usis ecx, qword ptr [edx - 1024]
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x6c,0x4a,0x80]
          vcvttsd2usis ecx, qword ptr [edx - 1024]

// CHECK: vcvttss2sis ecx, xmm2
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x6d,0xca]
          vcvttss2sis ecx, xmm2

// CHECK: vcvttss2sis ecx, xmm2, {sae}
// CHECK: encoding: [0x62,0xf5,0x7e,0x18,0x6d,0xca]
          vcvttss2sis ecx, xmm2, {sae}

// CHECK: vcvttss2sis ecx, dword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x6d,0x8c,0xf4,0x00,0x00,0x00,0x10]
          vcvttss2sis ecx, dword ptr [esp + 8*esi + 268435456]

// CHECK: vcvttss2sis ecx, dword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x6d,0x8c,0x87,0x23,0x01,0x00,0x00]
          vcvttss2sis ecx, dword ptr [edi + 4*eax + 291]

// CHECK: vcvttss2sis ecx, dword ptr [eax]
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x6d,0x08]
          vcvttss2sis ecx, dword ptr [eax]

// CHECK: vcvttss2sis ecx, dword ptr [2*ebp - 128]
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x6d,0x0c,0x6d,0x80,0xff,0xff,0xff]
          vcvttss2sis ecx, dword ptr [2*ebp - 128]

// CHECK: vcvttss2sis ecx, dword ptr [ecx + 508]
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x6d,0x49,0x7f]
          vcvttss2sis ecx, dword ptr [ecx + 508]

// CHECK: vcvttss2sis ecx, dword ptr [edx - 512]
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x6d,0x4a,0x80]
          vcvttss2sis ecx, dword ptr [edx - 512]

// CHECK: vcvttss2usis ecx, xmm2
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x6c,0xca]
          vcvttss2usis ecx, xmm2

// CHECK: vcvttss2usis ecx, xmm2, {sae}
// CHECK: encoding: [0x62,0xf5,0x7e,0x18,0x6c,0xca]
          vcvttss2usis ecx, xmm2, {sae}

// CHECK: vcvttss2usis ecx, dword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x6c,0x8c,0xf4,0x00,0x00,0x00,0x10]
          vcvttss2usis ecx, dword ptr [esp + 8*esi + 268435456]

// CHECK: vcvttss2usis ecx, dword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x6c,0x8c,0x87,0x23,0x01,0x00,0x00]
          vcvttss2usis ecx, dword ptr [edi + 4*eax + 291]

// CHECK: vcvttss2usis ecx, dword ptr [eax]
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x6c,0x08]
          vcvttss2usis ecx, dword ptr [eax]

// CHECK: vcvttss2usis ecx, dword ptr [2*ebp - 128]
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x6c,0x0c,0x6d,0x80,0xff,0xff,0xff]
          vcvttss2usis ecx, dword ptr [2*ebp - 128]

// CHECK: vcvttss2usis ecx, dword ptr [ecx + 508]
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x6c,0x49,0x7f]
          vcvttss2usis ecx, dword ptr [ecx + 508]

// CHECK: vcvttss2usis ecx, dword ptr [edx - 512]
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x6c,0x4a,0x80]
          vcvttss2usis ecx, dword ptr [edx - 512]

// CHECK: vcvttpd2dqs xmm2, xmm3
// CHECK: encoding: [0x62,0xf5,0xfc,0x08,0x6d,0xd3]
          vcvttpd2dqs xmm2, xmm3

// CHECK: vcvttpd2dqs xmm2 {k7}, xmm3
// CHECK: encoding: [0x62,0xf5,0xfc,0x0f,0x6d,0xd3]
          vcvttpd2dqs xmm2 {k7}, xmm3

// CHECK: vcvttpd2dqs xmm2 {k7} {z}, xmm3
// CHECK: encoding: [0x62,0xf5,0xfc,0x8f,0x6d,0xd3]
          vcvttpd2dqs xmm2 {k7} {z}, xmm3

// CHECK: vcvttpd2dqs xmm2, ymm3
// CHECK: encoding: [0x62,0xf5,0xfc,0x28,0x6d,0xd3]
          vcvttpd2dqs xmm2, ymm3

// CHECK: vcvttpd2dqs xmm2, ymm3, {sae}
// CHECK: encoding: [0x62,0xf5,0xf8,0x18,0x6d,0xd3]
          vcvttpd2dqs xmm2, ymm3, {sae}

// CHECK: vcvttpd2dqs xmm2 {k7}, ymm3
// CHECK: encoding: [0x62,0xf5,0xfc,0x2f,0x6d,0xd3]
          vcvttpd2dqs xmm2 {k7}, ymm3

// CHECK: vcvttpd2dqs xmm2 {k7} {z}, ymm3, {sae}
// CHECK: encoding: [0x62,0xf5,0xf8,0x9f,0x6d,0xd3]
          vcvttpd2dqs xmm2 {k7} {z}, ymm3, {sae}

// CHECK: vcvttpd2dqs ymm2, zmm3
// CHECK: encoding: [0x62,0xf5,0xfc,0x48,0x6d,0xd3]
          vcvttpd2dqs ymm2, zmm3

// CHECK: vcvttpd2dqs ymm2, zmm3, {sae}
// CHECK: encoding: [0x62,0xf5,0xfc,0x18,0x6d,0xd3]
          vcvttpd2dqs ymm2, zmm3, {sae}

// CHECK: vcvttpd2dqs ymm2 {k7}, zmm3
// CHECK: encoding: [0x62,0xf5,0xfc,0x4f,0x6d,0xd3]
          vcvttpd2dqs ymm2 {k7}, zmm3

// CHECK: vcvttpd2dqs ymm2 {k7} {z}, zmm3, {sae}
// CHECK: encoding: [0x62,0xf5,0xfc,0x9f,0x6d,0xd3]
          vcvttpd2dqs ymm2 {k7} {z}, zmm3, {sae}

// CHECK: vcvttpd2dqs xmm2, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0xfc,0x08,0x6d,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttpd2dqs xmm2, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvttpd2dqs xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0xfc,0x0f,0x6d,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttpd2dqs xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]

// CHECK: vcvttpd2dqs xmm2, qword ptr [eax]{1to2}
// CHECK: encoding: [0x62,0xf5,0xfc,0x18,0x6d,0x10]
          vcvttpd2dqs xmm2, qword ptr [eax]{1to2}

// CHECK: vcvttpd2dqs xmm2, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf5,0xfc,0x08,0x6d,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvttpd2dqs xmm2, xmmword ptr [2*ebp - 512]

// CHECK: vcvttpd2dqs xmm2 {k7} {z}, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf5,0xfc,0x8f,0x6d,0x51,0x7f]
          vcvttpd2dqs xmm2 {k7} {z}, xmmword ptr [ecx + 2032]

// CHECK: vcvttpd2dqs xmm2 {k7} {z}, qword ptr [edx - 1024]{1to2}
// CHECK: encoding: [0x62,0xf5,0xfc,0x9f,0x6d,0x52,0x80]
          vcvttpd2dqs xmm2 {k7} {z}, qword ptr [edx - 1024]{1to2}

// CHECK: vcvttpd2dqs xmm2, qword ptr [eax]{1to4}
// CHECK: encoding: [0x62,0xf5,0xfc,0x38,0x6d,0x10]
          vcvttpd2dqs xmm2, qword ptr [eax]{1to4}

// CHECK: vcvttpd2dqs xmm2, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf5,0xfc,0x28,0x6d,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvttpd2dqs xmm2, ymmword ptr [2*ebp - 1024]

// CHECK: vcvttpd2dqs xmm2 {k7} {z}, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf5,0xfc,0xaf,0x6d,0x51,0x7f]
          vcvttpd2dqs xmm2 {k7} {z}, ymmword ptr [ecx + 4064]

// CHECK: vcvttpd2dqs xmm2 {k7} {z}, qword ptr [edx - 1024]{1to4}
// CHECK: encoding: [0x62,0xf5,0xfc,0xbf,0x6d,0x52,0x80]
          vcvttpd2dqs xmm2 {k7} {z}, qword ptr [edx - 1024]{1to4}

// CHECK: vcvttpd2dqs ymm2, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0xfc,0x48,0x6d,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttpd2dqs ymm2, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvttpd2dqs ymm2 {k7}, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0xfc,0x4f,0x6d,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttpd2dqs ymm2 {k7}, zmmword ptr [edi + 4*eax + 291]

// CHECK: vcvttpd2dqs ymm2, qword ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf5,0xfc,0x58,0x6d,0x10]
          vcvttpd2dqs ymm2, qword ptr [eax]{1to8}

// CHECK: vcvttpd2dqs ymm2, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf5,0xfc,0x48,0x6d,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvttpd2dqs ymm2, zmmword ptr [2*ebp - 2048]

// CHECK: vcvttpd2dqs ymm2 {k7} {z}, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0xfc,0xcf,0x6d,0x51,0x7f]
          vcvttpd2dqs ymm2 {k7} {z}, zmmword ptr [ecx + 8128]

// CHECK: vcvttpd2dqs ymm2 {k7} {z}, qword ptr [edx - 1024]{1to8}
// CHECK: encoding: [0x62,0xf5,0xfc,0xdf,0x6d,0x52,0x80]
          vcvttpd2dqs ymm2 {k7} {z}, qword ptr [edx - 1024]{1to8}

// CHECK: vcvttpd2qqs xmm2, xmm3
// CHECK: encoding: [0x62,0xf5,0xfd,0x08,0x6d,0xd3]
          vcvttpd2qqs xmm2, xmm3

// CHECK: vcvttpd2qqs xmm2 {k7}, xmm3
// CHECK: encoding: [0x62,0xf5,0xfd,0x0f,0x6d,0xd3]
          vcvttpd2qqs xmm2 {k7}, xmm3

// CHECK: vcvttpd2qqs xmm2 {k7} {z}, xmm3
// CHECK: encoding: [0x62,0xf5,0xfd,0x8f,0x6d,0xd3]
          vcvttpd2qqs xmm2 {k7} {z}, xmm3

// CHECK: vcvttpd2qqs ymm2, ymm3
// CHECK: encoding: [0x62,0xf5,0xfd,0x28,0x6d,0xd3]
          vcvttpd2qqs ymm2, ymm3

// CHECK: vcvttpd2qqs ymm2, ymm3, {sae}
// CHECK: encoding: [0x62,0xf5,0xf9,0x18,0x6d,0xd3]
          vcvttpd2qqs ymm2, ymm3, {sae}

// CHECK: vcvttpd2qqs ymm2 {k7}, ymm3
// CHECK: encoding: [0x62,0xf5,0xfd,0x2f,0x6d,0xd3]
          vcvttpd2qqs ymm2 {k7}, ymm3

// CHECK: vcvttpd2qqs ymm2 {k7} {z}, ymm3, {sae}
// CHECK: encoding: [0x62,0xf5,0xf9,0x9f,0x6d,0xd3]
          vcvttpd2qqs ymm2 {k7} {z}, ymm3, {sae}

// CHECK: vcvttpd2qqs zmm2, zmm3
// CHECK: encoding: [0x62,0xf5,0xfd,0x48,0x6d,0xd3]
          vcvttpd2qqs zmm2, zmm3

// CHECK: vcvttpd2qqs zmm2, zmm3, {sae}
// CHECK: encoding: [0x62,0xf5,0xfd,0x18,0x6d,0xd3]
          vcvttpd2qqs zmm2, zmm3, {sae}

// CHECK: vcvttpd2qqs zmm2 {k7}, zmm3
// CHECK: encoding: [0x62,0xf5,0xfd,0x4f,0x6d,0xd3]
          vcvttpd2qqs zmm2 {k7}, zmm3

// CHECK: vcvttpd2qqs zmm2 {k7} {z}, zmm3, {sae}
// CHECK: encoding: [0x62,0xf5,0xfd,0x9f,0x6d,0xd3]
          vcvttpd2qqs zmm2 {k7} {z}, zmm3, {sae}

// CHECK: vcvttpd2qqs xmm2, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0xfd,0x08,0x6d,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttpd2qqs xmm2, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvttpd2qqs xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0xfd,0x0f,0x6d,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttpd2qqs xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]

// CHECK: vcvttpd2qqs xmm2, qword ptr [eax]{1to2}
// CHECK: encoding: [0x62,0xf5,0xfd,0x18,0x6d,0x10]
          vcvttpd2qqs xmm2, qword ptr [eax]{1to2}

// CHECK: vcvttpd2qqs xmm2, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf5,0xfd,0x08,0x6d,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvttpd2qqs xmm2, xmmword ptr [2*ebp - 512]

// CHECK: vcvttpd2qqs xmm2 {k7} {z}, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf5,0xfd,0x8f,0x6d,0x51,0x7f]
          vcvttpd2qqs xmm2 {k7} {z}, xmmword ptr [ecx + 2032]

// CHECK: vcvttpd2qqs xmm2 {k7} {z}, qword ptr [edx - 1024]{1to2}
// CHECK: encoding: [0x62,0xf5,0xfd,0x9f,0x6d,0x52,0x80]
          vcvttpd2qqs xmm2 {k7} {z}, qword ptr [edx - 1024]{1to2}

// CHECK: vcvttpd2qqs ymm2, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0xfd,0x28,0x6d,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttpd2qqs ymm2, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvttpd2qqs ymm2 {k7}, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0xfd,0x2f,0x6d,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttpd2qqs ymm2 {k7}, ymmword ptr [edi + 4*eax + 291]

// CHECK: vcvttpd2qqs ymm2, qword ptr [eax]{1to4}
// CHECK: encoding: [0x62,0xf5,0xfd,0x38,0x6d,0x10]
          vcvttpd2qqs ymm2, qword ptr [eax]{1to4}

// CHECK: vcvttpd2qqs ymm2, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf5,0xfd,0x28,0x6d,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvttpd2qqs ymm2, ymmword ptr [2*ebp - 1024]

// CHECK: vcvttpd2qqs ymm2 {k7} {z}, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf5,0xfd,0xaf,0x6d,0x51,0x7f]
          vcvttpd2qqs ymm2 {k7} {z}, ymmword ptr [ecx + 4064]

// CHECK: vcvttpd2qqs ymm2 {k7} {z}, qword ptr [edx - 1024]{1to4}
// CHECK: encoding: [0x62,0xf5,0xfd,0xbf,0x6d,0x52,0x80]
          vcvttpd2qqs ymm2 {k7} {z}, qword ptr [edx - 1024]{1to4}

// CHECK: vcvttpd2qqs zmm2, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0xfd,0x48,0x6d,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttpd2qqs zmm2, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvttpd2qqs zmm2 {k7}, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0xfd,0x4f,0x6d,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttpd2qqs zmm2 {k7}, zmmword ptr [edi + 4*eax + 291]

// CHECK: vcvttpd2qqs zmm2, qword ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf5,0xfd,0x58,0x6d,0x10]
          vcvttpd2qqs zmm2, qword ptr [eax]{1to8}

// CHECK: vcvttpd2qqs zmm2, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf5,0xfd,0x48,0x6d,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvttpd2qqs zmm2, zmmword ptr [2*ebp - 2048]

// CHECK: vcvttpd2qqs zmm2 {k7} {z}, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0xfd,0xcf,0x6d,0x51,0x7f]
          vcvttpd2qqs zmm2 {k7} {z}, zmmword ptr [ecx + 8128]

// CHECK: vcvttpd2qqs zmm2 {k7} {z}, qword ptr [edx - 1024]{1to8}
// CHECK: encoding: [0x62,0xf5,0xfd,0xdf,0x6d,0x52,0x80]
          vcvttpd2qqs zmm2 {k7} {z}, qword ptr [edx - 1024]{1to8}

// CHECK: vcvttpd2udqs xmm2, xmm3
// CHECK: encoding: [0x62,0xf5,0xfc,0x08,0x6c,0xd3]
          vcvttpd2udqs xmm2, xmm3

// CHECK: vcvttpd2udqs xmm2 {k7}, xmm3
// CHECK: encoding: [0x62,0xf5,0xfc,0x0f,0x6c,0xd3]
          vcvttpd2udqs xmm2 {k7}, xmm3

// CHECK: vcvttpd2udqs xmm2 {k7} {z}, xmm3
// CHECK: encoding: [0x62,0xf5,0xfc,0x8f,0x6c,0xd3]
          vcvttpd2udqs xmm2 {k7} {z}, xmm3

// CHECK: vcvttpd2udqs xmm2, ymm3
// CHECK: encoding: [0x62,0xf5,0xfc,0x28,0x6c,0xd3]
          vcvttpd2udqs xmm2, ymm3

// CHECK: vcvttpd2udqs xmm2, ymm3, {sae}
// CHECK: encoding: [0x62,0xf5,0xf8,0x18,0x6c,0xd3]
          vcvttpd2udqs xmm2, ymm3, {sae}

// CHECK: vcvttpd2udqs xmm2 {k7}, ymm3
// CHECK: encoding: [0x62,0xf5,0xfc,0x2f,0x6c,0xd3]
          vcvttpd2udqs xmm2 {k7}, ymm3

// CHECK: vcvttpd2udqs xmm2 {k7} {z}, ymm3, {sae}
// CHECK: encoding: [0x62,0xf5,0xf8,0x9f,0x6c,0xd3]
          vcvttpd2udqs xmm2 {k7} {z}, ymm3, {sae}

// CHECK: vcvttpd2udqs ymm2, zmm3
// CHECK: encoding: [0x62,0xf5,0xfc,0x48,0x6c,0xd3]
          vcvttpd2udqs ymm2, zmm3

// CHECK: vcvttpd2udqs ymm2, zmm3, {sae}
// CHECK: encoding: [0x62,0xf5,0xfc,0x18,0x6c,0xd3]
          vcvttpd2udqs ymm2, zmm3, {sae}

// CHECK: vcvttpd2udqs ymm2 {k7}, zmm3
// CHECK: encoding: [0x62,0xf5,0xfc,0x4f,0x6c,0xd3]
          vcvttpd2udqs ymm2 {k7}, zmm3

// CHECK: vcvttpd2udqs ymm2 {k7} {z}, zmm3, {sae}
// CHECK: encoding: [0x62,0xf5,0xfc,0x9f,0x6c,0xd3]
          vcvttpd2udqs ymm2 {k7} {z}, zmm3, {sae}

// CHECK: vcvttpd2udqs xmm2, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0xfc,0x08,0x6c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttpd2udqs xmm2, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvttpd2udqs xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0xfc,0x0f,0x6c,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttpd2udqs xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]

// CHECK: vcvttpd2udqs xmm2, qword ptr [eax]{1to2}
// CHECK: encoding: [0x62,0xf5,0xfc,0x18,0x6c,0x10]
          vcvttpd2udqs xmm2, qword ptr [eax]{1to2}

// CHECK: vcvttpd2udqs xmm2, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf5,0xfc,0x08,0x6c,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvttpd2udqs xmm2, xmmword ptr [2*ebp - 512]

// CHECK: vcvttpd2udqs xmm2 {k7} {z}, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf5,0xfc,0x8f,0x6c,0x51,0x7f]
          vcvttpd2udqs xmm2 {k7} {z}, xmmword ptr [ecx + 2032]

// CHECK: vcvttpd2udqs xmm2 {k7} {z}, qword ptr [edx - 1024]{1to2}
// CHECK: encoding: [0x62,0xf5,0xfc,0x9f,0x6c,0x52,0x80]
          vcvttpd2udqs xmm2 {k7} {z}, qword ptr [edx - 1024]{1to2}

// CHECK: vcvttpd2udqs xmm2, qword ptr [eax]{1to4}
// CHECK: encoding: [0x62,0xf5,0xfc,0x38,0x6c,0x10]
          vcvttpd2udqs xmm2, qword ptr [eax]{1to4}

// CHECK: vcvttpd2udqs xmm2, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf5,0xfc,0x28,0x6c,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvttpd2udqs xmm2, ymmword ptr [2*ebp - 1024]

// CHECK: vcvttpd2udqs xmm2 {k7} {z}, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf5,0xfc,0xaf,0x6c,0x51,0x7f]
          vcvttpd2udqs xmm2 {k7} {z}, ymmword ptr [ecx + 4064]

// CHECK: vcvttpd2udqs xmm2 {k7} {z}, qword ptr [edx - 1024]{1to4}
// CHECK: encoding: [0x62,0xf5,0xfc,0xbf,0x6c,0x52,0x80]
          vcvttpd2udqs xmm2 {k7} {z}, qword ptr [edx - 1024]{1to4}

// CHECK: vcvttpd2udqs ymm2, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0xfc,0x48,0x6c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttpd2udqs ymm2, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvttpd2udqs ymm2 {k7}, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0xfc,0x4f,0x6c,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttpd2udqs ymm2 {k7}, zmmword ptr [edi + 4*eax + 291]

// CHECK: vcvttpd2udqs ymm2, qword ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf5,0xfc,0x58,0x6c,0x10]
          vcvttpd2udqs ymm2, qword ptr [eax]{1to8}

// CHECK: vcvttpd2udqs ymm2, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf5,0xfc,0x48,0x6c,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvttpd2udqs ymm2, zmmword ptr [2*ebp - 2048]

// CHECK: vcvttpd2udqs ymm2 {k7} {z}, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0xfc,0xcf,0x6c,0x51,0x7f]
          vcvttpd2udqs ymm2 {k7} {z}, zmmword ptr [ecx + 8128]

// CHECK: vcvttpd2udqs ymm2 {k7} {z}, qword ptr [edx - 1024]{1to8}
// CHECK: encoding: [0x62,0xf5,0xfc,0xdf,0x6c,0x52,0x80]
          vcvttpd2udqs ymm2 {k7} {z}, qword ptr [edx - 1024]{1to8}

// CHECK: vcvttpd2uqqs xmm2, xmm3
// CHECK: encoding: [0x62,0xf5,0xfd,0x08,0x6c,0xd3]
          vcvttpd2uqqs xmm2, xmm3

// CHECK: vcvttpd2uqqs xmm2 {k7}, xmm3
// CHECK: encoding: [0x62,0xf5,0xfd,0x0f,0x6c,0xd3]
          vcvttpd2uqqs xmm2 {k7}, xmm3

// CHECK: vcvttpd2uqqs xmm2 {k7} {z}, xmm3
// CHECK: encoding: [0x62,0xf5,0xfd,0x8f,0x6c,0xd3]
          vcvttpd2uqqs xmm2 {k7} {z}, xmm3

// CHECK: vcvttpd2uqqs ymm2, ymm3
// CHECK: encoding: [0x62,0xf5,0xfd,0x28,0x6c,0xd3]
          vcvttpd2uqqs ymm2, ymm3

// CHECK: vcvttpd2uqqs ymm2, ymm3, {sae}
// CHECK: encoding: [0x62,0xf5,0xf9,0x18,0x6c,0xd3]
          vcvttpd2uqqs ymm2, ymm3, {sae}

// CHECK: vcvttpd2uqqs ymm2 {k7}, ymm3
// CHECK: encoding: [0x62,0xf5,0xfd,0x2f,0x6c,0xd3]
          vcvttpd2uqqs ymm2 {k7}, ymm3

// CHECK: vcvttpd2uqqs ymm2 {k7} {z}, ymm3, {sae}
// CHECK: encoding: [0x62,0xf5,0xf9,0x9f,0x6c,0xd3]
          vcvttpd2uqqs ymm2 {k7} {z}, ymm3, {sae}

// CHECK: vcvttpd2uqqs zmm2, zmm3
// CHECK: encoding: [0x62,0xf5,0xfd,0x48,0x6c,0xd3]
          vcvttpd2uqqs zmm2, zmm3

// CHECK: vcvttpd2uqqs zmm2, zmm3, {sae}
// CHECK: encoding: [0x62,0xf5,0xfd,0x18,0x6c,0xd3]
          vcvttpd2uqqs zmm2, zmm3, {sae}

// CHECK: vcvttpd2uqqs zmm2 {k7}, zmm3
// CHECK: encoding: [0x62,0xf5,0xfd,0x4f,0x6c,0xd3]
          vcvttpd2uqqs zmm2 {k7}, zmm3

// CHECK: vcvttpd2uqqs zmm2 {k7} {z}, zmm3, {sae}
// CHECK: encoding: [0x62,0xf5,0xfd,0x9f,0x6c,0xd3]
          vcvttpd2uqqs zmm2 {k7} {z}, zmm3, {sae}

// CHECK: vcvttpd2uqqs xmm2, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0xfd,0x08,0x6c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttpd2uqqs xmm2, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvttpd2uqqs xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0xfd,0x0f,0x6c,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttpd2uqqs xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]

// CHECK: vcvttpd2uqqs xmm2, qword ptr [eax]{1to2}
// CHECK: encoding: [0x62,0xf5,0xfd,0x18,0x6c,0x10]
          vcvttpd2uqqs xmm2, qword ptr [eax]{1to2}

// CHECK: vcvttpd2uqqs xmm2, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf5,0xfd,0x08,0x6c,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvttpd2uqqs xmm2, xmmword ptr [2*ebp - 512]

// CHECK: vcvttpd2uqqs xmm2 {k7} {z}, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf5,0xfd,0x8f,0x6c,0x51,0x7f]
          vcvttpd2uqqs xmm2 {k7} {z}, xmmword ptr [ecx + 2032]

// CHECK: vcvttpd2uqqs xmm2 {k7} {z}, qword ptr [edx - 1024]{1to2}
// CHECK: encoding: [0x62,0xf5,0xfd,0x9f,0x6c,0x52,0x80]
          vcvttpd2uqqs xmm2 {k7} {z}, qword ptr [edx - 1024]{1to2}

// CHECK: vcvttpd2uqqs ymm2, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0xfd,0x28,0x6c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttpd2uqqs ymm2, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvttpd2uqqs ymm2 {k7}, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0xfd,0x2f,0x6c,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttpd2uqqs ymm2 {k7}, ymmword ptr [edi + 4*eax + 291]

// CHECK: vcvttpd2uqqs ymm2, qword ptr [eax]{1to4}
// CHECK: encoding: [0x62,0xf5,0xfd,0x38,0x6c,0x10]
          vcvttpd2uqqs ymm2, qword ptr [eax]{1to4}

// CHECK: vcvttpd2uqqs ymm2, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf5,0xfd,0x28,0x6c,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvttpd2uqqs ymm2, ymmword ptr [2*ebp - 1024]

// CHECK: vcvttpd2uqqs ymm2 {k7} {z}, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf5,0xfd,0xaf,0x6c,0x51,0x7f]
          vcvttpd2uqqs ymm2 {k7} {z}, ymmword ptr [ecx + 4064]

// CHECK: vcvttpd2uqqs ymm2 {k7} {z}, qword ptr [edx - 1024]{1to4}
// CHECK: encoding: [0x62,0xf5,0xfd,0xbf,0x6c,0x52,0x80]
          vcvttpd2uqqs ymm2 {k7} {z}, qword ptr [edx - 1024]{1to4}

// CHECK: vcvttpd2uqqs zmm2, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0xfd,0x48,0x6c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttpd2uqqs zmm2, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvttpd2uqqs zmm2 {k7}, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0xfd,0x4f,0x6c,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttpd2uqqs zmm2 {k7}, zmmword ptr [edi + 4*eax + 291]

// CHECK: vcvttpd2uqqs zmm2, qword ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf5,0xfd,0x58,0x6c,0x10]
          vcvttpd2uqqs zmm2, qword ptr [eax]{1to8}

// CHECK: vcvttpd2uqqs zmm2, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf5,0xfd,0x48,0x6c,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvttpd2uqqs zmm2, zmmword ptr [2*ebp - 2048]

// CHECK: vcvttpd2uqqs zmm2 {k7} {z}, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0xfd,0xcf,0x6c,0x51,0x7f]
          vcvttpd2uqqs zmm2 {k7} {z}, zmmword ptr [ecx + 8128]

// CHECK: vcvttpd2uqqs zmm2 {k7} {z}, qword ptr [edx - 1024]{1to8}
// CHECK: encoding: [0x62,0xf5,0xfd,0xdf,0x6c,0x52,0x80]
          vcvttpd2uqqs zmm2 {k7} {z}, qword ptr [edx - 1024]{1to8}

// CHECK: vcvttps2dqs xmm2, xmm3
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x6d,0xd3]
          vcvttps2dqs xmm2, xmm3

// CHECK: vcvttps2dqs xmm2 {k7}, xmm3
// CHECK: encoding: [0x62,0xf5,0x7c,0x0f,0x6d,0xd3]
          vcvttps2dqs xmm2 {k7}, xmm3

// CHECK: vcvttps2dqs xmm2 {k7} {z}, xmm3
// CHECK: encoding: [0x62,0xf5,0x7c,0x8f,0x6d,0xd3]
          vcvttps2dqs xmm2 {k7} {z}, xmm3

// CHECK: vcvttps2dqs ymm2, ymm3
// CHECK: encoding: [0x62,0xf5,0x7c,0x28,0x6d,0xd3]
          vcvttps2dqs ymm2, ymm3

// CHECK: vcvttps2dqs ymm2, ymm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x78,0x18,0x6d,0xd3]
          vcvttps2dqs ymm2, ymm3, {sae}

// CHECK: vcvttps2dqs ymm2 {k7}, ymm3
// CHECK: encoding: [0x62,0xf5,0x7c,0x2f,0x6d,0xd3]
          vcvttps2dqs ymm2 {k7}, ymm3

// CHECK: vcvttps2dqs ymm2 {k7} {z}, ymm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x78,0x9f,0x6d,0xd3]
          vcvttps2dqs ymm2 {k7} {z}, ymm3, {sae}

// CHECK: vcvttps2dqs zmm2, zmm3
// CHECK: encoding: [0x62,0xf5,0x7c,0x48,0x6d,0xd3]
          vcvttps2dqs zmm2, zmm3

// CHECK: vcvttps2dqs zmm2, zmm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x7c,0x18,0x6d,0xd3]
          vcvttps2dqs zmm2, zmm3, {sae}

// CHECK: vcvttps2dqs zmm2 {k7}, zmm3
// CHECK: encoding: [0x62,0xf5,0x7c,0x4f,0x6d,0xd3]
          vcvttps2dqs zmm2 {k7}, zmm3

// CHECK: vcvttps2dqs zmm2 {k7} {z}, zmm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x7c,0x9f,0x6d,0xd3]
          vcvttps2dqs zmm2 {k7} {z}, zmm3, {sae}

// CHECK: vcvttps2dqs xmm2, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x6d,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttps2dqs xmm2, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvttps2dqs xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7c,0x0f,0x6d,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttps2dqs xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]

// CHECK: vcvttps2dqs xmm2, dword ptr [eax]{1to4}
// CHECK: encoding: [0x62,0xf5,0x7c,0x18,0x6d,0x10]
          vcvttps2dqs xmm2, dword ptr [eax]{1to4}

// CHECK: vcvttps2dqs xmm2, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x6d,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvttps2dqs xmm2, xmmword ptr [2*ebp - 512]

// CHECK: vcvttps2dqs xmm2 {k7} {z}, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf5,0x7c,0x8f,0x6d,0x51,0x7f]
          vcvttps2dqs xmm2 {k7} {z}, xmmword ptr [ecx + 2032]

// CHECK: vcvttps2dqs xmm2 {k7} {z}, dword ptr [edx - 512]{1to4}
// CHECK: encoding: [0x62,0xf5,0x7c,0x9f,0x6d,0x52,0x80]
          vcvttps2dqs xmm2 {k7} {z}, dword ptr [edx - 512]{1to4}

// CHECK: vcvttps2dqs ymm2, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7c,0x28,0x6d,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttps2dqs ymm2, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvttps2dqs ymm2 {k7}, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7c,0x2f,0x6d,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttps2dqs ymm2 {k7}, ymmword ptr [edi + 4*eax + 291]

// CHECK: vcvttps2dqs ymm2, dword ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7c,0x38,0x6d,0x10]
          vcvttps2dqs ymm2, dword ptr [eax]{1to8}

// CHECK: vcvttps2dqs ymm2, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf5,0x7c,0x28,0x6d,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvttps2dqs ymm2, ymmword ptr [2*ebp - 1024]

// CHECK: vcvttps2dqs ymm2 {k7} {z}, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf5,0x7c,0xaf,0x6d,0x51,0x7f]
          vcvttps2dqs ymm2 {k7} {z}, ymmword ptr [ecx + 4064]

// CHECK: vcvttps2dqs ymm2 {k7} {z}, dword ptr [edx - 512]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7c,0xbf,0x6d,0x52,0x80]
          vcvttps2dqs ymm2 {k7} {z}, dword ptr [edx - 512]{1to8}

// CHECK: vcvttps2dqs zmm2, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7c,0x48,0x6d,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttps2dqs zmm2, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvttps2dqs zmm2 {k7}, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7c,0x4f,0x6d,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttps2dqs zmm2 {k7}, zmmword ptr [edi + 4*eax + 291]

// CHECK: vcvttps2dqs zmm2, dword ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7c,0x58,0x6d,0x10]
          vcvttps2dqs zmm2, dword ptr [eax]{1to16}

// CHECK: vcvttps2dqs zmm2, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf5,0x7c,0x48,0x6d,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvttps2dqs zmm2, zmmword ptr [2*ebp - 2048]

// CHECK: vcvttps2dqs zmm2 {k7} {z}, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0x7c,0xcf,0x6d,0x51,0x7f]
          vcvttps2dqs zmm2 {k7} {z}, zmmword ptr [ecx + 8128]

// CHECK: vcvttps2dqs zmm2 {k7} {z}, dword ptr [edx - 512]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7c,0xdf,0x6d,0x52,0x80]
          vcvttps2dqs zmm2 {k7} {z}, dword ptr [edx - 512]{1to16}

// CHECK: vcvttps2qqs xmm2, xmm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x6d,0xd3]
          vcvttps2qqs xmm2, xmm3

// CHECK: vcvttps2qqs xmm2 {k7}, xmm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x0f,0x6d,0xd3]
          vcvttps2qqs xmm2 {k7}, xmm3

// CHECK: vcvttps2qqs xmm2 {k7} {z}, xmm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x8f,0x6d,0xd3]
          vcvttps2qqs xmm2 {k7} {z}, xmm3

// CHECK: vcvttps2qqs ymm2, xmm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x6d,0xd3]
          vcvttps2qqs ymm2, xmm3

// CHECK: vcvttps2qqs ymm2, xmm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x79,0x18,0x6d,0xd3]
          vcvttps2qqs ymm2, xmm3, {sae}

// CHECK: vcvttps2qqs ymm2 {k7}, xmm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x2f,0x6d,0xd3]
          vcvttps2qqs ymm2 {k7}, xmm3

// CHECK: vcvttps2qqs ymm2 {k7} {z}, xmm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x79,0x9f,0x6d,0xd3]
          vcvttps2qqs ymm2 {k7} {z}, xmm3, {sae}

// CHECK: vcvttps2qqs zmm2, ymm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x6d,0xd3]
          vcvttps2qqs zmm2, ymm3

// CHECK: vcvttps2qqs zmm2, ymm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x7d,0x18,0x6d,0xd3]
          vcvttps2qqs zmm2, ymm3, {sae}

// CHECK: vcvttps2qqs zmm2 {k7}, ymm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x4f,0x6d,0xd3]
          vcvttps2qqs zmm2 {k7}, ymm3

// CHECK: vcvttps2qqs zmm2 {k7} {z}, ymm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x7d,0x9f,0x6d,0xd3]
          vcvttps2qqs zmm2 {k7} {z}, ymm3, {sae}

// CHECK: vcvttps2qqs xmm2, qword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x6d,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttps2qqs xmm2, qword ptr [esp + 8*esi + 268435456]

// CHECK: vcvttps2qqs xmm2 {k7}, qword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7d,0x0f,0x6d,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttps2qqs xmm2 {k7}, qword ptr [edi + 4*eax + 291]

// CHECK: vcvttps2qqs xmm2, dword ptr [eax]{1to2}
// CHECK: encoding: [0x62,0xf5,0x7d,0x18,0x6d,0x10]
          vcvttps2qqs xmm2, dword ptr [eax]{1to2}

// CHECK: vcvttps2qqs xmm2, qword ptr [2*ebp - 256]
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x6d,0x14,0x6d,0x00,0xff,0xff,0xff]
          vcvttps2qqs xmm2, qword ptr [2*ebp - 256]

// CHECK: vcvttps2qqs xmm2 {k7} {z}, qword ptr [ecx + 1016]
// CHECK: encoding: [0x62,0xf5,0x7d,0x8f,0x6d,0x51,0x7f]
          vcvttps2qqs xmm2 {k7} {z}, qword ptr [ecx + 1016]

// CHECK: vcvttps2qqs xmm2 {k7} {z}, dword ptr [edx - 512]{1to2}
// CHECK: encoding: [0x62,0xf5,0x7d,0x9f,0x6d,0x52,0x80]
          vcvttps2qqs xmm2 {k7} {z}, dword ptr [edx - 512]{1to2}

// CHECK: vcvttps2qqs ymm2, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x6d,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttps2qqs ymm2, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvttps2qqs ymm2 {k7}, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7d,0x2f,0x6d,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttps2qqs ymm2 {k7}, xmmword ptr [edi + 4*eax + 291]

// CHECK: vcvttps2qqs ymm2, dword ptr [eax]{1to4}
// CHECK: encoding: [0x62,0xf5,0x7d,0x38,0x6d,0x10]
          vcvttps2qqs ymm2, dword ptr [eax]{1to4}

// CHECK: vcvttps2qqs ymm2, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x6d,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvttps2qqs ymm2, xmmword ptr [2*ebp - 512]

// CHECK: vcvttps2qqs ymm2 {k7} {z}, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf5,0x7d,0xaf,0x6d,0x51,0x7f]
          vcvttps2qqs ymm2 {k7} {z}, xmmword ptr [ecx + 2032]

// CHECK: vcvttps2qqs ymm2 {k7} {z}, dword ptr [edx - 512]{1to4}
// CHECK: encoding: [0x62,0xf5,0x7d,0xbf,0x6d,0x52,0x80]
          vcvttps2qqs ymm2 {k7} {z}, dword ptr [edx - 512]{1to4}

// CHECK: vcvttps2qqs zmm2, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x6d,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttps2qqs zmm2, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvttps2qqs zmm2 {k7}, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7d,0x4f,0x6d,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttps2qqs zmm2 {k7}, ymmword ptr [edi + 4*eax + 291]

// CHECK: vcvttps2qqs zmm2, dword ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7d,0x58,0x6d,0x10]
          vcvttps2qqs zmm2, dword ptr [eax]{1to8}

// CHECK: vcvttps2qqs zmm2, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x6d,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvttps2qqs zmm2, ymmword ptr [2*ebp - 1024]

// CHECK: vcvttps2qqs zmm2 {k7} {z}, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf5,0x7d,0xcf,0x6d,0x51,0x7f]
          vcvttps2qqs zmm2 {k7} {z}, ymmword ptr [ecx + 4064]

// CHECK: vcvttps2qqs zmm2 {k7} {z}, dword ptr [edx - 512]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7d,0xdf,0x6d,0x52,0x80]
          vcvttps2qqs zmm2 {k7} {z}, dword ptr [edx - 512]{1to8}

// CHECK: vcvttps2udqs xmm2, xmm3
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x6c,0xd3]
          vcvttps2udqs xmm2, xmm3

// CHECK: vcvttps2udqs xmm2 {k7}, xmm3
// CHECK: encoding: [0x62,0xf5,0x7c,0x0f,0x6c,0xd3]
          vcvttps2udqs xmm2 {k7}, xmm3

// CHECK: vcvttps2udqs xmm2 {k7} {z}, xmm3
// CHECK: encoding: [0x62,0xf5,0x7c,0x8f,0x6c,0xd3]
          vcvttps2udqs xmm2 {k7} {z}, xmm3

// CHECK: vcvttps2udqs ymm2, ymm3
// CHECK: encoding: [0x62,0xf5,0x7c,0x28,0x6c,0xd3]
          vcvttps2udqs ymm2, ymm3

// CHECK: vcvttps2udqs ymm2, ymm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x78,0x18,0x6c,0xd3]
          vcvttps2udqs ymm2, ymm3, {sae}

// CHECK: vcvttps2udqs ymm2 {k7}, ymm3
// CHECK: encoding: [0x62,0xf5,0x7c,0x2f,0x6c,0xd3]
          vcvttps2udqs ymm2 {k7}, ymm3

// CHECK: vcvttps2udqs ymm2 {k7} {z}, ymm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x78,0x9f,0x6c,0xd3]
          vcvttps2udqs ymm2 {k7} {z}, ymm3, {sae}

// CHECK: vcvttps2udqs zmm2, zmm3
// CHECK: encoding: [0x62,0xf5,0x7c,0x48,0x6c,0xd3]
          vcvttps2udqs zmm2, zmm3

// CHECK: vcvttps2udqs zmm2, zmm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x7c,0x18,0x6c,0xd3]
          vcvttps2udqs zmm2, zmm3, {sae}

// CHECK: vcvttps2udqs zmm2 {k7}, zmm3
// CHECK: encoding: [0x62,0xf5,0x7c,0x4f,0x6c,0xd3]
          vcvttps2udqs zmm2 {k7}, zmm3

// CHECK: vcvttps2udqs zmm2 {k7} {z}, zmm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x7c,0x9f,0x6c,0xd3]
          vcvttps2udqs zmm2 {k7} {z}, zmm3, {sae}

// CHECK: vcvttps2udqs xmm2, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x6c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttps2udqs xmm2, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvttps2udqs xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7c,0x0f,0x6c,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttps2udqs xmm2 {k7}, xmmword ptr [edi + 4*eax + 291]

// CHECK: vcvttps2udqs xmm2, dword ptr [eax]{1to4}
// CHECK: encoding: [0x62,0xf5,0x7c,0x18,0x6c,0x10]
          vcvttps2udqs xmm2, dword ptr [eax]{1to4}

// CHECK: vcvttps2udqs xmm2, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x6c,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvttps2udqs xmm2, xmmword ptr [2*ebp - 512]

// CHECK: vcvttps2udqs xmm2 {k7} {z}, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf5,0x7c,0x8f,0x6c,0x51,0x7f]
          vcvttps2udqs xmm2 {k7} {z}, xmmword ptr [ecx + 2032]

// CHECK: vcvttps2udqs xmm2 {k7} {z}, dword ptr [edx - 512]{1to4}
// CHECK: encoding: [0x62,0xf5,0x7c,0x9f,0x6c,0x52,0x80]
          vcvttps2udqs xmm2 {k7} {z}, dword ptr [edx - 512]{1to4}

// CHECK: vcvttps2udqs ymm2, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7c,0x28,0x6c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttps2udqs ymm2, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvttps2udqs ymm2 {k7}, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7c,0x2f,0x6c,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttps2udqs ymm2 {k7}, ymmword ptr [edi + 4*eax + 291]

// CHECK: vcvttps2udqs ymm2, dword ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7c,0x38,0x6c,0x10]
          vcvttps2udqs ymm2, dword ptr [eax]{1to8}

// CHECK: vcvttps2udqs ymm2, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf5,0x7c,0x28,0x6c,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvttps2udqs ymm2, ymmword ptr [2*ebp - 1024]

// CHECK: vcvttps2udqs ymm2 {k7} {z}, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf5,0x7c,0xaf,0x6c,0x51,0x7f]
          vcvttps2udqs ymm2 {k7} {z}, ymmword ptr [ecx + 4064]

// CHECK: vcvttps2udqs ymm2 {k7} {z}, dword ptr [edx - 512]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7c,0xbf,0x6c,0x52,0x80]
          vcvttps2udqs ymm2 {k7} {z}, dword ptr [edx - 512]{1to8}

// CHECK: vcvttps2udqs zmm2, zmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7c,0x48,0x6c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttps2udqs zmm2, zmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvttps2udqs zmm2 {k7}, zmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7c,0x4f,0x6c,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttps2udqs zmm2 {k7}, zmmword ptr [edi + 4*eax + 291]

// CHECK: vcvttps2udqs zmm2, dword ptr [eax]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7c,0x58,0x6c,0x10]
          vcvttps2udqs zmm2, dword ptr [eax]{1to16}

// CHECK: vcvttps2udqs zmm2, zmmword ptr [2*ebp - 2048]
// CHECK: encoding: [0x62,0xf5,0x7c,0x48,0x6c,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvttps2udqs zmm2, zmmword ptr [2*ebp - 2048]

// CHECK: vcvttps2udqs zmm2 {k7} {z}, zmmword ptr [ecx + 8128]
// CHECK: encoding: [0x62,0xf5,0x7c,0xcf,0x6c,0x51,0x7f]
          vcvttps2udqs zmm2 {k7} {z}, zmmword ptr [ecx + 8128]

// CHECK: vcvttps2udqs zmm2 {k7} {z}, dword ptr [edx - 512]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7c,0xdf,0x6c,0x52,0x80]
          vcvttps2udqs zmm2 {k7} {z}, dword ptr [edx - 512]{1to16}

// CHECK: vcvttps2uqqs xmm2, xmm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x6c,0xd3]
          vcvttps2uqqs xmm2, xmm3

// CHECK: vcvttps2uqqs xmm2 {k7}, xmm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x0f,0x6c,0xd3]
          vcvttps2uqqs xmm2 {k7}, xmm3

// CHECK: vcvttps2uqqs xmm2 {k7} {z}, xmm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x8f,0x6c,0xd3]
          vcvttps2uqqs xmm2 {k7} {z}, xmm3

// CHECK: vcvttps2uqqs ymm2, xmm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x6c,0xd3]
          vcvttps2uqqs ymm2, xmm3

// CHECK: vcvttps2uqqs ymm2, xmm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x79,0x18,0x6c,0xd3]
          vcvttps2uqqs ymm2, xmm3, {sae}

// CHECK: vcvttps2uqqs ymm2 {k7}, xmm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x2f,0x6c,0xd3]
          vcvttps2uqqs ymm2 {k7}, xmm3

// CHECK: vcvttps2uqqs ymm2 {k7} {z}, xmm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x79,0x9f,0x6c,0xd3]
          vcvttps2uqqs ymm2 {k7} {z}, xmm3, {sae}

// CHECK: vcvttps2uqqs zmm2, ymm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x6c,0xd3]
          vcvttps2uqqs zmm2, ymm3

// CHECK: vcvttps2uqqs zmm2, ymm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x7d,0x18,0x6c,0xd3]
          vcvttps2uqqs zmm2, ymm3, {sae}

// CHECK: vcvttps2uqqs zmm2 {k7}, ymm3
// CHECK: encoding: [0x62,0xf5,0x7d,0x4f,0x6c,0xd3]
          vcvttps2uqqs zmm2 {k7}, ymm3

// CHECK: vcvttps2uqqs zmm2 {k7} {z}, ymm3, {sae}
// CHECK: encoding: [0x62,0xf5,0x7d,0x9f,0x6c,0xd3]
          vcvttps2uqqs zmm2 {k7} {z}, ymm3, {sae}

// CHECK: vcvttps2uqqs xmm2, qword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x6c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttps2uqqs xmm2, qword ptr [esp + 8*esi + 268435456]

// CHECK: vcvttps2uqqs xmm2 {k7}, qword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7d,0x0f,0x6c,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttps2uqqs xmm2 {k7}, qword ptr [edi + 4*eax + 291]

// CHECK: vcvttps2uqqs xmm2, dword ptr [eax]{1to2}
// CHECK: encoding: [0x62,0xf5,0x7d,0x18,0x6c,0x10]
          vcvttps2uqqs xmm2, dword ptr [eax]{1to2}

// CHECK: vcvttps2uqqs xmm2, qword ptr [2*ebp - 256]
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x6c,0x14,0x6d,0x00,0xff,0xff,0xff]
          vcvttps2uqqs xmm2, qword ptr [2*ebp - 256]

// CHECK: vcvttps2uqqs xmm2 {k7} {z}, qword ptr [ecx + 1016]
// CHECK: encoding: [0x62,0xf5,0x7d,0x8f,0x6c,0x51,0x7f]
          vcvttps2uqqs xmm2 {k7} {z}, qword ptr [ecx + 1016]

// CHECK: vcvttps2uqqs xmm2 {k7} {z}, dword ptr [edx - 512]{1to2}
// CHECK: encoding: [0x62,0xf5,0x7d,0x9f,0x6c,0x52,0x80]
          vcvttps2uqqs xmm2 {k7} {z}, dword ptr [edx - 512]{1to2}

// CHECK: vcvttps2uqqs ymm2, xmmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x6c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttps2uqqs ymm2, xmmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvttps2uqqs ymm2 {k7}, xmmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7d,0x2f,0x6c,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttps2uqqs ymm2 {k7}, xmmword ptr [edi + 4*eax + 291]

// CHECK: vcvttps2uqqs ymm2, dword ptr [eax]{1to4}
// CHECK: encoding: [0x62,0xf5,0x7d,0x38,0x6c,0x10]
          vcvttps2uqqs ymm2, dword ptr [eax]{1to4}

// CHECK: vcvttps2uqqs ymm2, xmmword ptr [2*ebp - 512]
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x6c,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvttps2uqqs ymm2, xmmword ptr [2*ebp - 512]

// CHECK: vcvttps2uqqs ymm2 {k7} {z}, xmmword ptr [ecx + 2032]
// CHECK: encoding: [0x62,0xf5,0x7d,0xaf,0x6c,0x51,0x7f]
          vcvttps2uqqs ymm2 {k7} {z}, xmmword ptr [ecx + 2032]

// CHECK: vcvttps2uqqs ymm2 {k7} {z}, dword ptr [edx - 512]{1to4}
// CHECK: encoding: [0x62,0xf5,0x7d,0xbf,0x6c,0x52,0x80]
          vcvttps2uqqs ymm2 {k7} {z}, dword ptr [edx - 512]{1to4}

// CHECK: vcvttps2uqqs zmm2, ymmword ptr [esp + 8*esi + 268435456]
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x6c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttps2uqqs zmm2, ymmword ptr [esp + 8*esi + 268435456]

// CHECK: vcvttps2uqqs zmm2 {k7}, ymmword ptr [edi + 4*eax + 291]
// CHECK: encoding: [0x62,0xf5,0x7d,0x4f,0x6c,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttps2uqqs zmm2 {k7}, ymmword ptr [edi + 4*eax + 291]

// CHECK: vcvttps2uqqs zmm2, dword ptr [eax]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7d,0x58,0x6c,0x10]
          vcvttps2uqqs zmm2, dword ptr [eax]{1to8}

// CHECK: vcvttps2uqqs zmm2, ymmword ptr [2*ebp - 1024]
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x6c,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvttps2uqqs zmm2, ymmword ptr [2*ebp - 1024]

// CHECK: vcvttps2uqqs zmm2 {k7} {z}, ymmword ptr [ecx + 4064]
// CHECK: encoding: [0x62,0xf5,0x7d,0xcf,0x6c,0x51,0x7f]
          vcvttps2uqqs zmm2 {k7} {z}, ymmword ptr [ecx + 4064]

// CHECK: vcvttps2uqqs zmm2 {k7} {z}, dword ptr [edx - 512]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7d,0xdf,0x6c,0x52,0x80]
          vcvttps2uqqs zmm2 {k7} {z}, dword ptr [edx - 512]{1to8}

