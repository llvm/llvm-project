// RUN: llvm-mc -triple x86_64 -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

// CHECK: vcvttsd2sis ecx, xmm22
// CHECK: encoding: [0x62,0xb5,0x7f,0x08,0x6d,0xce]
          vcvttsd2sis ecx, xmm22

// CHECK: vcvttsd2sis ecx, xmm22, {sae}
// CHECK: encoding: [0x62,0xb5,0x7f,0x18,0x6d,0xce]
          vcvttsd2sis ecx, xmm22, {sae}

// CHECK: vcvttsd2sis r9, xmm22
// CHECK: encoding: [0x62,0x35,0xff,0x08,0x6d,0xce]
          vcvttsd2sis r9, xmm22

// CHECK: vcvttsd2sis r9, xmm22, {sae}
// CHECK: encoding: [0x62,0x35,0xff,0x18,0x6d,0xce]
          vcvttsd2sis r9, xmm22, {sae}

// CHECK: vcvttsd2sis ecx, qword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xb5,0x7f,0x08,0x6d,0x8c,0xf5,0x00,0x00,0x00,0x10]
          vcvttsd2sis ecx, qword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttsd2sis ecx, qword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xd5,0x7f,0x08,0x6d,0x8c,0x80,0x23,0x01,0x00,0x00]
          vcvttsd2sis ecx, qword ptr [r8 + 4*rax + 291]

// CHECK: vcvttsd2sis ecx, qword ptr [rip]
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x6d,0x0d,0x00,0x00,0x00,0x00]
          vcvttsd2sis ecx, qword ptr [rip]

// CHECK: vcvttsd2sis ecx, qword ptr [2*rbp - 256]
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x6d,0x0c,0x6d,0x00,0xff,0xff,0xff]
          vcvttsd2sis ecx, qword ptr [2*rbp - 256]

// CHECK: vcvttsd2sis ecx, qword ptr [rcx + 1016]
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x6d,0x49,0x7f]
          vcvttsd2sis ecx, qword ptr [rcx + 1016]

// CHECK: vcvttsd2sis ecx, qword ptr [rdx - 1024]
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x6d,0x4a,0x80]
          vcvttsd2sis ecx, qword ptr [rdx - 1024]

// CHECK: vcvttsd2sis r9, qword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x35,0xff,0x08,0x6d,0x8c,0xf5,0x00,0x00,0x00,0x10]
          vcvttsd2sis r9, qword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttsd2sis r9, qword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0x55,0xff,0x08,0x6d,0x8c,0x80,0x23,0x01,0x00,0x00]
          vcvttsd2sis r9, qword ptr [r8 + 4*rax + 291]

// CHECK: vcvttsd2sis r9, qword ptr [rip]
// CHECK: encoding: [0x62,0x75,0xff,0x08,0x6d,0x0d,0x00,0x00,0x00,0x00]
          vcvttsd2sis r9, qword ptr [rip]

// CHECK: vcvttsd2sis r9, qword ptr [2*rbp - 256]
// CHECK: encoding: [0x62,0x75,0xff,0x08,0x6d,0x0c,0x6d,0x00,0xff,0xff,0xff]
          vcvttsd2sis r9, qword ptr [2*rbp - 256]

// CHECK: vcvttsd2sis r9, qword ptr [rcx + 1016]
// CHECK: encoding: [0x62,0x75,0xff,0x08,0x6d,0x49,0x7f]
          vcvttsd2sis r9, qword ptr [rcx + 1016]

// CHECK: vcvttsd2sis r9, qword ptr [rdx - 1024]
// CHECK: encoding: [0x62,0x75,0xff,0x08,0x6d,0x4a,0x80]
          vcvttsd2sis r9, qword ptr [rdx - 1024]

// CHECK: vcvttsd2usis ecx, xmm22
// CHECK: encoding: [0x62,0xb5,0x7f,0x08,0x6c,0xce]
          vcvttsd2usis ecx, xmm22

// CHECK: vcvttsd2usis ecx, xmm22, {sae}
// CHECK: encoding: [0x62,0xb5,0x7f,0x18,0x6c,0xce]
          vcvttsd2usis ecx, xmm22, {sae}

// CHECK: vcvttsd2usis r9, xmm22
// CHECK: encoding: [0x62,0x35,0xff,0x08,0x6c,0xce]
          vcvttsd2usis r9, xmm22

// CHECK: vcvttsd2usis r9, xmm22, {sae}
// CHECK: encoding: [0x62,0x35,0xff,0x18,0x6c,0xce]
          vcvttsd2usis r9, xmm22, {sae}

// CHECK: vcvttsd2usis ecx, qword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xb5,0x7f,0x08,0x6c,0x8c,0xf5,0x00,0x00,0x00,0x10]
          vcvttsd2usis ecx, qword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttsd2usis ecx, qword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xd5,0x7f,0x08,0x6c,0x8c,0x80,0x23,0x01,0x00,0x00]
          vcvttsd2usis ecx, qword ptr [r8 + 4*rax + 291]

// CHECK: vcvttsd2usis ecx, qword ptr [rip]
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x6c,0x0d,0x00,0x00,0x00,0x00]
          vcvttsd2usis ecx, qword ptr [rip]

// CHECK: vcvttsd2usis ecx, qword ptr [2*rbp - 256]
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x6c,0x0c,0x6d,0x00,0xff,0xff,0xff]
          vcvttsd2usis ecx, qword ptr [2*rbp - 256]

// CHECK: vcvttsd2usis ecx, qword ptr [rcx + 1016]
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x6c,0x49,0x7f]
          vcvttsd2usis ecx, qword ptr [rcx + 1016]

// CHECK: vcvttsd2usis ecx, qword ptr [rdx - 1024]
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x6c,0x4a,0x80]
          vcvttsd2usis ecx, qword ptr [rdx - 1024]

// CHECK: vcvttsd2usis r9, qword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x35,0xff,0x08,0x6c,0x8c,0xf5,0x00,0x00,0x00,0x10]
          vcvttsd2usis r9, qword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttsd2usis r9, qword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0x55,0xff,0x08,0x6c,0x8c,0x80,0x23,0x01,0x00,0x00]
          vcvttsd2usis r9, qword ptr [r8 + 4*rax + 291]

// CHECK: vcvttsd2usis r9, qword ptr [rip]
// CHECK: encoding: [0x62,0x75,0xff,0x08,0x6c,0x0d,0x00,0x00,0x00,0x00]
          vcvttsd2usis r9, qword ptr [rip]

// CHECK: vcvttsd2usis r9, qword ptr [2*rbp - 256]
// CHECK: encoding: [0x62,0x75,0xff,0x08,0x6c,0x0c,0x6d,0x00,0xff,0xff,0xff]
          vcvttsd2usis r9, qword ptr [2*rbp - 256]

// CHECK: vcvttsd2usis r9, qword ptr [rcx + 1016]
// CHECK: encoding: [0x62,0x75,0xff,0x08,0x6c,0x49,0x7f]
          vcvttsd2usis r9, qword ptr [rcx + 1016]

// CHECK: vcvttsd2usis r9, qword ptr [rdx - 1024]
// CHECK: encoding: [0x62,0x75,0xff,0x08,0x6c,0x4a,0x80]
          vcvttsd2usis r9, qword ptr [rdx - 1024]

// CHECK: vcvttss2sis ecx, xmm22
// CHECK: encoding: [0x62,0xb5,0x7e,0x08,0x6d,0xce]
          vcvttss2sis ecx, xmm22

// CHECK: vcvttss2sis ecx, xmm22, {sae}
// CHECK: encoding: [0x62,0xb5,0x7e,0x18,0x6d,0xce]
          vcvttss2sis ecx, xmm22, {sae}

// CHECK: vcvttss2sis r9, xmm22
// CHECK: encoding: [0x62,0x35,0xfe,0x08,0x6d,0xce]
          vcvttss2sis r9, xmm22

// CHECK: vcvttss2sis r9, xmm22, {sae}
// CHECK: encoding: [0x62,0x35,0xfe,0x18,0x6d,0xce]
          vcvttss2sis r9, xmm22, {sae}

// CHECK: vcvttss2sis ecx, dword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xb5,0x7e,0x08,0x6d,0x8c,0xf5,0x00,0x00,0x00,0x10]
          vcvttss2sis ecx, dword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttss2sis ecx, dword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xd5,0x7e,0x08,0x6d,0x8c,0x80,0x23,0x01,0x00,0x00]
          vcvttss2sis ecx, dword ptr [r8 + 4*rax + 291]

// CHECK: vcvttss2sis ecx, dword ptr [rip]
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x6d,0x0d,0x00,0x00,0x00,0x00]
          vcvttss2sis ecx, dword ptr [rip]

// CHECK: vcvttss2sis ecx, dword ptr [2*rbp - 128]
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x6d,0x0c,0x6d,0x80,0xff,0xff,0xff]
          vcvttss2sis ecx, dword ptr [2*rbp - 128]

// CHECK: vcvttss2sis ecx, dword ptr [rcx + 508]
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x6d,0x49,0x7f]
          vcvttss2sis ecx, dword ptr [rcx + 508]

// CHECK: vcvttss2sis ecx, dword ptr [rdx - 512]
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x6d,0x4a,0x80]
          vcvttss2sis ecx, dword ptr [rdx - 512]

// CHECK: vcvttss2sis r9, dword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x35,0xfe,0x08,0x6d,0x8c,0xf5,0x00,0x00,0x00,0x10]
          vcvttss2sis r9, dword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttss2sis r9, dword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0x55,0xfe,0x08,0x6d,0x8c,0x80,0x23,0x01,0x00,0x00]
          vcvttss2sis r9, dword ptr [r8 + 4*rax + 291]

// CHECK: vcvttss2sis r9, dword ptr [rip]
// CHECK: encoding: [0x62,0x75,0xfe,0x08,0x6d,0x0d,0x00,0x00,0x00,0x00]
          vcvttss2sis r9, dword ptr [rip]

// CHECK: vcvttss2sis r9, dword ptr [2*rbp - 128]
// CHECK: encoding: [0x62,0x75,0xfe,0x08,0x6d,0x0c,0x6d,0x80,0xff,0xff,0xff]
          vcvttss2sis r9, dword ptr [2*rbp - 128]

// CHECK: vcvttss2sis r9, dword ptr [rcx + 508]
// CHECK: encoding: [0x62,0x75,0xfe,0x08,0x6d,0x49,0x7f]
          vcvttss2sis r9, dword ptr [rcx + 508]

// CHECK: vcvttss2sis r9, dword ptr [rdx - 512]
// CHECK: encoding: [0x62,0x75,0xfe,0x08,0x6d,0x4a,0x80]
          vcvttss2sis r9, dword ptr [rdx - 512]

// CHECK: vcvttss2usis ecx, xmm22
// CHECK: encoding: [0x62,0xb5,0x7e,0x08,0x6c,0xce]
          vcvttss2usis ecx, xmm22

// CHECK: vcvttss2usis ecx, xmm22, {sae}
// CHECK: encoding: [0x62,0xb5,0x7e,0x18,0x6c,0xce]
          vcvttss2usis ecx, xmm22, {sae}

// CHECK: vcvttss2usis r9, xmm22
// CHECK: encoding: [0x62,0x35,0xfe,0x08,0x6c,0xce]
          vcvttss2usis r9, xmm22

// CHECK: vcvttss2usis r9, xmm22, {sae}
// CHECK: encoding: [0x62,0x35,0xfe,0x18,0x6c,0xce]
          vcvttss2usis r9, xmm22, {sae}

// CHECK: vcvttss2usis ecx, dword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xb5,0x7e,0x08,0x6c,0x8c,0xf5,0x00,0x00,0x00,0x10]
          vcvttss2usis ecx, dword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttss2usis ecx, dword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xd5,0x7e,0x08,0x6c,0x8c,0x80,0x23,0x01,0x00,0x00]
          vcvttss2usis ecx, dword ptr [r8 + 4*rax + 291]

// CHECK: vcvttss2usis ecx, dword ptr [rip]
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x6c,0x0d,0x00,0x00,0x00,0x00]
          vcvttss2usis ecx, dword ptr [rip]

// CHECK: vcvttss2usis ecx, dword ptr [2*rbp - 128]
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x6c,0x0c,0x6d,0x80,0xff,0xff,0xff]
          vcvttss2usis ecx, dword ptr [2*rbp - 128]

// CHECK: vcvttss2usis ecx, dword ptr [rcx + 508]
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x6c,0x49,0x7f]
          vcvttss2usis ecx, dword ptr [rcx + 508]

// CHECK: vcvttss2usis ecx, dword ptr [rdx - 512]
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x6c,0x4a,0x80]
          vcvttss2usis ecx, dword ptr [rdx - 512]

// CHECK: vcvttss2usis r9, dword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0x35,0xfe,0x08,0x6c,0x8c,0xf5,0x00,0x00,0x00,0x10]
          vcvttss2usis r9, dword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttss2usis r9, dword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0x55,0xfe,0x08,0x6c,0x8c,0x80,0x23,0x01,0x00,0x00]
          vcvttss2usis r9, dword ptr [r8 + 4*rax + 291]

// CHECK: vcvttss2usis r9, dword ptr [rip]
// CHECK: encoding: [0x62,0x75,0xfe,0x08,0x6c,0x0d,0x00,0x00,0x00,0x00]
          vcvttss2usis r9, dword ptr [rip]

// CHECK: vcvttss2usis r9, dword ptr [2*rbp - 128]
// CHECK: encoding: [0x62,0x75,0xfe,0x08,0x6c,0x0c,0x6d,0x80,0xff,0xff,0xff]
          vcvttss2usis r9, dword ptr [2*rbp - 128]

// CHECK: vcvttss2usis r9, dword ptr [rcx + 508]
// CHECK: encoding: [0x62,0x75,0xfe,0x08,0x6c,0x49,0x7f]
          vcvttss2usis r9, dword ptr [rcx + 508]

// CHECK: vcvttss2usis r9, dword ptr [rdx - 512]
// CHECK: encoding: [0x62,0x75,0xfe,0x08,0x6c,0x4a,0x80]
          vcvttss2usis r9, dword ptr [rdx - 512]

// CHECK: vcvttpd2dqs xmm22, xmm23
// CHECK: encoding: [0x62,0xa5,0xfc,0x08,0x6d,0xf7]
          vcvttpd2dqs xmm22, xmm23

// CHECK: vcvttpd2dqs xmm22 {k7}, xmm23
// CHECK: encoding: [0x62,0xa5,0xfc,0x0f,0x6d,0xf7]
          vcvttpd2dqs xmm22 {k7}, xmm23

// CHECK: vcvttpd2dqs xmm22 {k7} {z}, xmm23
// CHECK: encoding: [0x62,0xa5,0xfc,0x8f,0x6d,0xf7]
          vcvttpd2dqs xmm22 {k7} {z}, xmm23

// CHECK: vcvttpd2dqs xmm22, ymm23
// CHECK: encoding: [0x62,0xa5,0xfc,0x28,0x6d,0xf7]
          vcvttpd2dqs xmm22, ymm23

// CHECK: vcvttpd2dqs xmm22, ymm23, {sae}
// CHECK: encoding: [0x62,0xa5,0xf8,0x18,0x6d,0xf7]
          vcvttpd2dqs xmm22, ymm23, {sae}

// CHECK: vcvttpd2dqs xmm22 {k7}, ymm23
// CHECK: encoding: [0x62,0xa5,0xfc,0x2f,0x6d,0xf7]
          vcvttpd2dqs xmm22 {k7}, ymm23

// CHECK: vcvttpd2dqs xmm22 {k7} {z}, ymm23, {sae}
// CHECK: encoding: [0x62,0xa5,0xf8,0x9f,0x6d,0xf7]
          vcvttpd2dqs xmm22 {k7} {z}, ymm23, {sae}

// CHECK: vcvttpd2dqs ymm22, zmm23
// CHECK: encoding: [0x62,0xa5,0xfc,0x48,0x6d,0xf7]
          vcvttpd2dqs ymm22, zmm23

// CHECK: vcvttpd2dqs ymm22, zmm23, {sae}
// CHECK: encoding: [0x62,0xa5,0xfc,0x18,0x6d,0xf7]
          vcvttpd2dqs ymm22, zmm23, {sae}

// CHECK: vcvttpd2dqs ymm22 {k7}, zmm23
// CHECK: encoding: [0x62,0xa5,0xfc,0x4f,0x6d,0xf7]
          vcvttpd2dqs ymm22 {k7}, zmm23

// CHECK: vcvttpd2dqs ymm22 {k7} {z}, zmm23, {sae}
// CHECK: encoding: [0x62,0xa5,0xfc,0x9f,0x6d,0xf7]
          vcvttpd2dqs ymm22 {k7} {z}, zmm23, {sae}

// CHECK: vcvttpd2dqs xmm22, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0xfc,0x08,0x6d,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvttpd2dqs xmm22, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttpd2dqs xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0xfc,0x0f,0x6d,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvttpd2dqs xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvttpd2dqs xmm22, qword ptr [rip]{1to2}
// CHECK: encoding: [0x62,0xe5,0xfc,0x18,0x6d,0x35,0x00,0x00,0x00,0x00]
          vcvttpd2dqs xmm22, qword ptr [rip]{1to2}

// CHECK: vcvttpd2dqs xmm22, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe5,0xfc,0x08,0x6d,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vcvttpd2dqs xmm22, xmmword ptr [2*rbp - 512]

// CHECK: vcvttpd2dqs xmm22 {k7} {z}, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe5,0xfc,0x8f,0x6d,0x71,0x7f]
          vcvttpd2dqs xmm22 {k7} {z}, xmmword ptr [rcx + 2032]

// CHECK: vcvttpd2dqs xmm22 {k7} {z}, qword ptr [rdx - 1024]{1to2}
// CHECK: encoding: [0x62,0xe5,0xfc,0x9f,0x6d,0x72,0x80]
          vcvttpd2dqs xmm22 {k7} {z}, qword ptr [rdx - 1024]{1to2}

// CHECK: vcvttpd2dqs xmm22, qword ptr [rip]{1to4}
// CHECK: encoding: [0x62,0xe5,0xfc,0x38,0x6d,0x35,0x00,0x00,0x00,0x00]
          vcvttpd2dqs xmm22, qword ptr [rip]{1to4}

// CHECK: vcvttpd2dqs xmm22, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe5,0xfc,0x28,0x6d,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vcvttpd2dqs xmm22, ymmword ptr [2*rbp - 1024]

// CHECK: vcvttpd2dqs xmm22 {k7} {z}, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe5,0xfc,0xaf,0x6d,0x71,0x7f]
          vcvttpd2dqs xmm22 {k7} {z}, ymmword ptr [rcx + 4064]

// CHECK: vcvttpd2dqs xmm22 {k7} {z}, qword ptr [rdx - 1024]{1to4}
// CHECK: encoding: [0x62,0xe5,0xfc,0xbf,0x6d,0x72,0x80]
          vcvttpd2dqs xmm22 {k7} {z}, qword ptr [rdx - 1024]{1to4}

// CHECK: vcvttpd2dqs ymm22, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0xfc,0x48,0x6d,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvttpd2dqs ymm22, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttpd2dqs ymm22 {k7}, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0xfc,0x4f,0x6d,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvttpd2dqs ymm22 {k7}, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvttpd2dqs ymm22, qword ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe5,0xfc,0x58,0x6d,0x35,0x00,0x00,0x00,0x00]
          vcvttpd2dqs ymm22, qword ptr [rip]{1to8}

// CHECK: vcvttpd2dqs ymm22, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe5,0xfc,0x48,0x6d,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vcvttpd2dqs ymm22, zmmword ptr [2*rbp - 2048]

// CHECK: vcvttpd2dqs ymm22 {k7} {z}, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe5,0xfc,0xcf,0x6d,0x71,0x7f]
          vcvttpd2dqs ymm22 {k7} {z}, zmmword ptr [rcx + 8128]

// CHECK: vcvttpd2dqs ymm22 {k7} {z}, qword ptr [rdx - 1024]{1to8}
// CHECK: encoding: [0x62,0xe5,0xfc,0xdf,0x6d,0x72,0x80]
          vcvttpd2dqs ymm22 {k7} {z}, qword ptr [rdx - 1024]{1to8}

// CHECK: vcvttpd2qqs xmm22, xmm23
// CHECK: encoding: [0x62,0xa5,0xfd,0x08,0x6d,0xf7]
          vcvttpd2qqs xmm22, xmm23

// CHECK: vcvttpd2qqs xmm22 {k7}, xmm23
// CHECK: encoding: [0x62,0xa5,0xfd,0x0f,0x6d,0xf7]
          vcvttpd2qqs xmm22 {k7}, xmm23

// CHECK: vcvttpd2qqs xmm22 {k7} {z}, xmm23
// CHECK: encoding: [0x62,0xa5,0xfd,0x8f,0x6d,0xf7]
          vcvttpd2qqs xmm22 {k7} {z}, xmm23

// CHECK: vcvttpd2qqs ymm22, ymm23
// CHECK: encoding: [0x62,0xa5,0xfd,0x28,0x6d,0xf7]
          vcvttpd2qqs ymm22, ymm23

// CHECK: vcvttpd2qqs ymm22, ymm23, {sae}
// CHECK: encoding: [0x62,0xa5,0xf9,0x18,0x6d,0xf7]
          vcvttpd2qqs ymm22, ymm23, {sae}

// CHECK: vcvttpd2qqs ymm22 {k7}, ymm23
// CHECK: encoding: [0x62,0xa5,0xfd,0x2f,0x6d,0xf7]
          vcvttpd2qqs ymm22 {k7}, ymm23

// CHECK: vcvttpd2qqs ymm22 {k7} {z}, ymm23, {sae}
// CHECK: encoding: [0x62,0xa5,0xf9,0x9f,0x6d,0xf7]
          vcvttpd2qqs ymm22 {k7} {z}, ymm23, {sae}

// CHECK: vcvttpd2qqs zmm22, zmm23
// CHECK: encoding: [0x62,0xa5,0xfd,0x48,0x6d,0xf7]
          vcvttpd2qqs zmm22, zmm23

// CHECK: vcvttpd2qqs zmm22, zmm23, {sae}
// CHECK: encoding: [0x62,0xa5,0xfd,0x18,0x6d,0xf7]
          vcvttpd2qqs zmm22, zmm23, {sae}

// CHECK: vcvttpd2qqs zmm22 {k7}, zmm23
// CHECK: encoding: [0x62,0xa5,0xfd,0x4f,0x6d,0xf7]
          vcvttpd2qqs zmm22 {k7}, zmm23

// CHECK: vcvttpd2qqs zmm22 {k7} {z}, zmm23, {sae}
// CHECK: encoding: [0x62,0xa5,0xfd,0x9f,0x6d,0xf7]
          vcvttpd2qqs zmm22 {k7} {z}, zmm23, {sae}

// CHECK: vcvttpd2qqs xmm22, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0xfd,0x08,0x6d,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvttpd2qqs xmm22, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttpd2qqs xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0xfd,0x0f,0x6d,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvttpd2qqs xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvttpd2qqs xmm22, qword ptr [rip]{1to2}
// CHECK: encoding: [0x62,0xe5,0xfd,0x18,0x6d,0x35,0x00,0x00,0x00,0x00]
          vcvttpd2qqs xmm22, qword ptr [rip]{1to2}

// CHECK: vcvttpd2qqs xmm22, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe5,0xfd,0x08,0x6d,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vcvttpd2qqs xmm22, xmmword ptr [2*rbp - 512]

// CHECK: vcvttpd2qqs xmm22 {k7} {z}, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe5,0xfd,0x8f,0x6d,0x71,0x7f]
          vcvttpd2qqs xmm22 {k7} {z}, xmmword ptr [rcx + 2032]

// CHECK: vcvttpd2qqs xmm22 {k7} {z}, qword ptr [rdx - 1024]{1to2}
// CHECK: encoding: [0x62,0xe5,0xfd,0x9f,0x6d,0x72,0x80]
          vcvttpd2qqs xmm22 {k7} {z}, qword ptr [rdx - 1024]{1to2}

// CHECK: vcvttpd2qqs ymm22, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0xfd,0x28,0x6d,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvttpd2qqs ymm22, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttpd2qqs ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0xfd,0x2f,0x6d,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvttpd2qqs ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vcvttpd2qqs ymm22, qword ptr [rip]{1to4}
// CHECK: encoding: [0x62,0xe5,0xfd,0x38,0x6d,0x35,0x00,0x00,0x00,0x00]
          vcvttpd2qqs ymm22, qword ptr [rip]{1to4}

// CHECK: vcvttpd2qqs ymm22, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe5,0xfd,0x28,0x6d,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vcvttpd2qqs ymm22, ymmword ptr [2*rbp - 1024]

// CHECK: vcvttpd2qqs ymm22 {k7} {z}, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe5,0xfd,0xaf,0x6d,0x71,0x7f]
          vcvttpd2qqs ymm22 {k7} {z}, ymmword ptr [rcx + 4064]

// CHECK: vcvttpd2qqs ymm22 {k7} {z}, qword ptr [rdx - 1024]{1to4}
// CHECK: encoding: [0x62,0xe5,0xfd,0xbf,0x6d,0x72,0x80]
          vcvttpd2qqs ymm22 {k7} {z}, qword ptr [rdx - 1024]{1to4}

// CHECK: vcvttpd2qqs zmm22, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0xfd,0x48,0x6d,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvttpd2qqs zmm22, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttpd2qqs zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0xfd,0x4f,0x6d,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvttpd2qqs zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvttpd2qqs zmm22, qword ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe5,0xfd,0x58,0x6d,0x35,0x00,0x00,0x00,0x00]
          vcvttpd2qqs zmm22, qword ptr [rip]{1to8}

// CHECK: vcvttpd2qqs zmm22, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe5,0xfd,0x48,0x6d,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vcvttpd2qqs zmm22, zmmword ptr [2*rbp - 2048]

// CHECK: vcvttpd2qqs zmm22 {k7} {z}, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe5,0xfd,0xcf,0x6d,0x71,0x7f]
          vcvttpd2qqs zmm22 {k7} {z}, zmmword ptr [rcx + 8128]

// CHECK: vcvttpd2qqs zmm22 {k7} {z}, qword ptr [rdx - 1024]{1to8}
// CHECK: encoding: [0x62,0xe5,0xfd,0xdf,0x6d,0x72,0x80]
          vcvttpd2qqs zmm22 {k7} {z}, qword ptr [rdx - 1024]{1to8}

// CHECK: vcvttpd2udqs xmm22, xmm23
// CHECK: encoding: [0x62,0xa5,0xfc,0x08,0x6c,0xf7]
          vcvttpd2udqs xmm22, xmm23

// CHECK: vcvttpd2udqs xmm22 {k7}, xmm23
// CHECK: encoding: [0x62,0xa5,0xfc,0x0f,0x6c,0xf7]
          vcvttpd2udqs xmm22 {k7}, xmm23

// CHECK: vcvttpd2udqs xmm22 {k7} {z}, xmm23
// CHECK: encoding: [0x62,0xa5,0xfc,0x8f,0x6c,0xf7]
          vcvttpd2udqs xmm22 {k7} {z}, xmm23

// CHECK: vcvttpd2udqs xmm22, ymm23
// CHECK: encoding: [0x62,0xa5,0xfc,0x28,0x6c,0xf7]
          vcvttpd2udqs xmm22, ymm23

// CHECK: vcvttpd2udqs xmm22, ymm23, {sae}
// CHECK: encoding: [0x62,0xa5,0xf8,0x18,0x6c,0xf7]
          vcvttpd2udqs xmm22, ymm23, {sae}

// CHECK: vcvttpd2udqs xmm22 {k7}, ymm23
// CHECK: encoding: [0x62,0xa5,0xfc,0x2f,0x6c,0xf7]
          vcvttpd2udqs xmm22 {k7}, ymm23

// CHECK: vcvttpd2udqs xmm22 {k7} {z}, ymm23, {sae}
// CHECK: encoding: [0x62,0xa5,0xf8,0x9f,0x6c,0xf7]
          vcvttpd2udqs xmm22 {k7} {z}, ymm23, {sae}

// CHECK: vcvttpd2udqs ymm22, zmm23
// CHECK: encoding: [0x62,0xa5,0xfc,0x48,0x6c,0xf7]
          vcvttpd2udqs ymm22, zmm23

// CHECK: vcvttpd2udqs ymm22, zmm23, {sae}
// CHECK: encoding: [0x62,0xa5,0xfc,0x18,0x6c,0xf7]
          vcvttpd2udqs ymm22, zmm23, {sae}

// CHECK: vcvttpd2udqs ymm22 {k7}, zmm23
// CHECK: encoding: [0x62,0xa5,0xfc,0x4f,0x6c,0xf7]
          vcvttpd2udqs ymm22 {k7}, zmm23

// CHECK: vcvttpd2udqs ymm22 {k7} {z}, zmm23, {sae}
// CHECK: encoding: [0x62,0xa5,0xfc,0x9f,0x6c,0xf7]
          vcvttpd2udqs ymm22 {k7} {z}, zmm23, {sae}

// CHECK: vcvttpd2udqs xmm22, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0xfc,0x08,0x6c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvttpd2udqs xmm22, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttpd2udqs xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0xfc,0x0f,0x6c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvttpd2udqs xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvttpd2udqs xmm22, qword ptr [rip]{1to2}
// CHECK: encoding: [0x62,0xe5,0xfc,0x18,0x6c,0x35,0x00,0x00,0x00,0x00]
          vcvttpd2udqs xmm22, qword ptr [rip]{1to2}

// CHECK: vcvttpd2udqs xmm22, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe5,0xfc,0x08,0x6c,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vcvttpd2udqs xmm22, xmmword ptr [2*rbp - 512]

// CHECK: vcvttpd2udqs xmm22 {k7} {z}, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe5,0xfc,0x8f,0x6c,0x71,0x7f]
          vcvttpd2udqs xmm22 {k7} {z}, xmmword ptr [rcx + 2032]

// CHECK: vcvttpd2udqs xmm22 {k7} {z}, qword ptr [rdx - 1024]{1to2}
// CHECK: encoding: [0x62,0xe5,0xfc,0x9f,0x6c,0x72,0x80]
          vcvttpd2udqs xmm22 {k7} {z}, qword ptr [rdx - 1024]{1to2}

// CHECK: vcvttpd2udqs xmm22, qword ptr [rip]{1to4}
// CHECK: encoding: [0x62,0xe5,0xfc,0x38,0x6c,0x35,0x00,0x00,0x00,0x00]
          vcvttpd2udqs xmm22, qword ptr [rip]{1to4}

// CHECK: vcvttpd2udqs xmm22, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe5,0xfc,0x28,0x6c,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vcvttpd2udqs xmm22, ymmword ptr [2*rbp - 1024]

// CHECK: vcvttpd2udqs xmm22 {k7} {z}, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe5,0xfc,0xaf,0x6c,0x71,0x7f]
          vcvttpd2udqs xmm22 {k7} {z}, ymmword ptr [rcx + 4064]

// CHECK: vcvttpd2udqs xmm22 {k7} {z}, qword ptr [rdx - 1024]{1to4}
// CHECK: encoding: [0x62,0xe5,0xfc,0xbf,0x6c,0x72,0x80]
          vcvttpd2udqs xmm22 {k7} {z}, qword ptr [rdx - 1024]{1to4}

// CHECK: vcvttpd2udqs ymm22, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0xfc,0x48,0x6c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvttpd2udqs ymm22, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttpd2udqs ymm22 {k7}, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0xfc,0x4f,0x6c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvttpd2udqs ymm22 {k7}, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvttpd2udqs ymm22, qword ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe5,0xfc,0x58,0x6c,0x35,0x00,0x00,0x00,0x00]
          vcvttpd2udqs ymm22, qword ptr [rip]{1to8}

// CHECK: vcvttpd2udqs ymm22, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe5,0xfc,0x48,0x6c,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vcvttpd2udqs ymm22, zmmword ptr [2*rbp - 2048]

// CHECK: vcvttpd2udqs ymm22 {k7} {z}, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe5,0xfc,0xcf,0x6c,0x71,0x7f]
          vcvttpd2udqs ymm22 {k7} {z}, zmmword ptr [rcx + 8128]

// CHECK: vcvttpd2udqs ymm22 {k7} {z}, qword ptr [rdx - 1024]{1to8}
// CHECK: encoding: [0x62,0xe5,0xfc,0xdf,0x6c,0x72,0x80]
          vcvttpd2udqs ymm22 {k7} {z}, qword ptr [rdx - 1024]{1to8}

// CHECK: vcvttpd2uqqs xmm22, xmm23
// CHECK: encoding: [0x62,0xa5,0xfd,0x08,0x6c,0xf7]
          vcvttpd2uqqs xmm22, xmm23

// CHECK: vcvttpd2uqqs xmm22 {k7}, xmm23
// CHECK: encoding: [0x62,0xa5,0xfd,0x0f,0x6c,0xf7]
          vcvttpd2uqqs xmm22 {k7}, xmm23

// CHECK: vcvttpd2uqqs xmm22 {k7} {z}, xmm23
// CHECK: encoding: [0x62,0xa5,0xfd,0x8f,0x6c,0xf7]
          vcvttpd2uqqs xmm22 {k7} {z}, xmm23

// CHECK: vcvttpd2uqqs ymm22, ymm23
// CHECK: encoding: [0x62,0xa5,0xfd,0x28,0x6c,0xf7]
          vcvttpd2uqqs ymm22, ymm23

// CHECK: vcvttpd2uqqs ymm22, ymm23, {sae}
// CHECK: encoding: [0x62,0xa5,0xf9,0x18,0x6c,0xf7]
          vcvttpd2uqqs ymm22, ymm23, {sae}

// CHECK: vcvttpd2uqqs ymm22 {k7}, ymm23
// CHECK: encoding: [0x62,0xa5,0xfd,0x2f,0x6c,0xf7]
          vcvttpd2uqqs ymm22 {k7}, ymm23

// CHECK: vcvttpd2uqqs ymm22 {k7} {z}, ymm23, {sae}
// CHECK: encoding: [0x62,0xa5,0xf9,0x9f,0x6c,0xf7]
          vcvttpd2uqqs ymm22 {k7} {z}, ymm23, {sae}

// CHECK: vcvttpd2uqqs zmm22, zmm23
// CHECK: encoding: [0x62,0xa5,0xfd,0x48,0x6c,0xf7]
          vcvttpd2uqqs zmm22, zmm23

// CHECK: vcvttpd2uqqs zmm22, zmm23, {sae}
// CHECK: encoding: [0x62,0xa5,0xfd,0x18,0x6c,0xf7]
          vcvttpd2uqqs zmm22, zmm23, {sae}

// CHECK: vcvttpd2uqqs zmm22 {k7}, zmm23
// CHECK: encoding: [0x62,0xa5,0xfd,0x4f,0x6c,0xf7]
          vcvttpd2uqqs zmm22 {k7}, zmm23

// CHECK: vcvttpd2uqqs zmm22 {k7} {z}, zmm23, {sae}
// CHECK: encoding: [0x62,0xa5,0xfd,0x9f,0x6c,0xf7]
          vcvttpd2uqqs zmm22 {k7} {z}, zmm23, {sae}

// CHECK: vcvttpd2uqqs xmm22, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0xfd,0x08,0x6c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvttpd2uqqs xmm22, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttpd2uqqs xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0xfd,0x0f,0x6c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvttpd2uqqs xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvttpd2uqqs xmm22, qword ptr [rip]{1to2}
// CHECK: encoding: [0x62,0xe5,0xfd,0x18,0x6c,0x35,0x00,0x00,0x00,0x00]
          vcvttpd2uqqs xmm22, qword ptr [rip]{1to2}

// CHECK: vcvttpd2uqqs xmm22, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe5,0xfd,0x08,0x6c,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vcvttpd2uqqs xmm22, xmmword ptr [2*rbp - 512]

// CHECK: vcvttpd2uqqs xmm22 {k7} {z}, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe5,0xfd,0x8f,0x6c,0x71,0x7f]
          vcvttpd2uqqs xmm22 {k7} {z}, xmmword ptr [rcx + 2032]

// CHECK: vcvttpd2uqqs xmm22 {k7} {z}, qword ptr [rdx - 1024]{1to2}
// CHECK: encoding: [0x62,0xe5,0xfd,0x9f,0x6c,0x72,0x80]
          vcvttpd2uqqs xmm22 {k7} {z}, qword ptr [rdx - 1024]{1to2}

// CHECK: vcvttpd2uqqs ymm22, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0xfd,0x28,0x6c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvttpd2uqqs ymm22, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttpd2uqqs ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0xfd,0x2f,0x6c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvttpd2uqqs ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vcvttpd2uqqs ymm22, qword ptr [rip]{1to4}
// CHECK: encoding: [0x62,0xe5,0xfd,0x38,0x6c,0x35,0x00,0x00,0x00,0x00]
          vcvttpd2uqqs ymm22, qword ptr [rip]{1to4}

// CHECK: vcvttpd2uqqs ymm22, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe5,0xfd,0x28,0x6c,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vcvttpd2uqqs ymm22, ymmword ptr [2*rbp - 1024]

// CHECK: vcvttpd2uqqs ymm22 {k7} {z}, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe5,0xfd,0xaf,0x6c,0x71,0x7f]
          vcvttpd2uqqs ymm22 {k7} {z}, ymmword ptr [rcx + 4064]

// CHECK: vcvttpd2uqqs ymm22 {k7} {z}, qword ptr [rdx - 1024]{1to4}
// CHECK: encoding: [0x62,0xe5,0xfd,0xbf,0x6c,0x72,0x80]
          vcvttpd2uqqs ymm22 {k7} {z}, qword ptr [rdx - 1024]{1to4}

// CHECK: vcvttpd2uqqs zmm22, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0xfd,0x48,0x6c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvttpd2uqqs zmm22, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttpd2uqqs zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0xfd,0x4f,0x6c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvttpd2uqqs zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvttpd2uqqs zmm22, qword ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe5,0xfd,0x58,0x6c,0x35,0x00,0x00,0x00,0x00]
          vcvttpd2uqqs zmm22, qword ptr [rip]{1to8}

// CHECK: vcvttpd2uqqs zmm22, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe5,0xfd,0x48,0x6c,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vcvttpd2uqqs zmm22, zmmword ptr [2*rbp - 2048]

// CHECK: vcvttpd2uqqs zmm22 {k7} {z}, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe5,0xfd,0xcf,0x6c,0x71,0x7f]
          vcvttpd2uqqs zmm22 {k7} {z}, zmmword ptr [rcx + 8128]

// CHECK: vcvttpd2uqqs zmm22 {k7} {z}, qword ptr [rdx - 1024]{1to8}
// CHECK: encoding: [0x62,0xe5,0xfd,0xdf,0x6c,0x72,0x80]
          vcvttpd2uqqs zmm22 {k7} {z}, qword ptr [rdx - 1024]{1to8}

// CHECK: vcvttps2dqs xmm22, xmm23
// CHECK: encoding: [0x62,0xa5,0x7c,0x08,0x6d,0xf7]
          vcvttps2dqs xmm22, xmm23

// CHECK: vcvttps2dqs xmm22 {k7}, xmm23
// CHECK: encoding: [0x62,0xa5,0x7c,0x0f,0x6d,0xf7]
          vcvttps2dqs xmm22 {k7}, xmm23

// CHECK: vcvttps2dqs xmm22 {k7} {z}, xmm23
// CHECK: encoding: [0x62,0xa5,0x7c,0x8f,0x6d,0xf7]
          vcvttps2dqs xmm22 {k7} {z}, xmm23

// CHECK: vcvttps2dqs ymm22, ymm23
// CHECK: encoding: [0x62,0xa5,0x7c,0x28,0x6d,0xf7]
          vcvttps2dqs ymm22, ymm23

// CHECK: vcvttps2dqs ymm22, ymm23, {sae}
// CHECK: encoding: [0x62,0xa5,0x78,0x18,0x6d,0xf7]
          vcvttps2dqs ymm22, ymm23, {sae}

// CHECK: vcvttps2dqs ymm22 {k7}, ymm23
// CHECK: encoding: [0x62,0xa5,0x7c,0x2f,0x6d,0xf7]
          vcvttps2dqs ymm22 {k7}, ymm23

// CHECK: vcvttps2dqs ymm22 {k7} {z}, ymm23, {sae}
// CHECK: encoding: [0x62,0xa5,0x78,0x9f,0x6d,0xf7]
          vcvttps2dqs ymm22 {k7} {z}, ymm23, {sae}

// CHECK: vcvttps2dqs zmm22, zmm23
// CHECK: encoding: [0x62,0xa5,0x7c,0x48,0x6d,0xf7]
          vcvttps2dqs zmm22, zmm23

// CHECK: vcvttps2dqs zmm22, zmm23, {sae}
// CHECK: encoding: [0x62,0xa5,0x7c,0x18,0x6d,0xf7]
          vcvttps2dqs zmm22, zmm23, {sae}

// CHECK: vcvttps2dqs zmm22 {k7}, zmm23
// CHECK: encoding: [0x62,0xa5,0x7c,0x4f,0x6d,0xf7]
          vcvttps2dqs zmm22 {k7}, zmm23

// CHECK: vcvttps2dqs zmm22 {k7} {z}, zmm23, {sae}
// CHECK: encoding: [0x62,0xa5,0x7c,0x9f,0x6d,0xf7]
          vcvttps2dqs zmm22 {k7} {z}, zmm23, {sae}

// CHECK: vcvttps2dqs xmm22, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7c,0x08,0x6d,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvttps2dqs xmm22, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttps2dqs xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7c,0x0f,0x6d,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvttps2dqs xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvttps2dqs xmm22, dword ptr [rip]{1to4}
// CHECK: encoding: [0x62,0xe5,0x7c,0x18,0x6d,0x35,0x00,0x00,0x00,0x00]
          vcvttps2dqs xmm22, dword ptr [rip]{1to4}

// CHECK: vcvttps2dqs xmm22, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe5,0x7c,0x08,0x6d,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vcvttps2dqs xmm22, xmmword ptr [2*rbp - 512]

// CHECK: vcvttps2dqs xmm22 {k7} {z}, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe5,0x7c,0x8f,0x6d,0x71,0x7f]
          vcvttps2dqs xmm22 {k7} {z}, xmmword ptr [rcx + 2032]

// CHECK: vcvttps2dqs xmm22 {k7} {z}, dword ptr [rdx - 512]{1to4}
// CHECK: encoding: [0x62,0xe5,0x7c,0x9f,0x6d,0x72,0x80]
          vcvttps2dqs xmm22 {k7} {z}, dword ptr [rdx - 512]{1to4}

// CHECK: vcvttps2dqs ymm22, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7c,0x28,0x6d,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvttps2dqs ymm22, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttps2dqs ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7c,0x2f,0x6d,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvttps2dqs ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vcvttps2dqs ymm22, dword ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe5,0x7c,0x38,0x6d,0x35,0x00,0x00,0x00,0x00]
          vcvttps2dqs ymm22, dword ptr [rip]{1to8}

// CHECK: vcvttps2dqs ymm22, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe5,0x7c,0x28,0x6d,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vcvttps2dqs ymm22, ymmword ptr [2*rbp - 1024]

// CHECK: vcvttps2dqs ymm22 {k7} {z}, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe5,0x7c,0xaf,0x6d,0x71,0x7f]
          vcvttps2dqs ymm22 {k7} {z}, ymmword ptr [rcx + 4064]

// CHECK: vcvttps2dqs ymm22 {k7} {z}, dword ptr [rdx - 512]{1to8}
// CHECK: encoding: [0x62,0xe5,0x7c,0xbf,0x6d,0x72,0x80]
          vcvttps2dqs ymm22 {k7} {z}, dword ptr [rdx - 512]{1to8}

// CHECK: vcvttps2dqs zmm22, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7c,0x48,0x6d,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvttps2dqs zmm22, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttps2dqs zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7c,0x4f,0x6d,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvttps2dqs zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvttps2dqs zmm22, dword ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe5,0x7c,0x58,0x6d,0x35,0x00,0x00,0x00,0x00]
          vcvttps2dqs zmm22, dword ptr [rip]{1to16}

// CHECK: vcvttps2dqs zmm22, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe5,0x7c,0x48,0x6d,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vcvttps2dqs zmm22, zmmword ptr [2*rbp - 2048]

// CHECK: vcvttps2dqs zmm22 {k7} {z}, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe5,0x7c,0xcf,0x6d,0x71,0x7f]
          vcvttps2dqs zmm22 {k7} {z}, zmmword ptr [rcx + 8128]

// CHECK: vcvttps2dqs zmm22 {k7} {z}, dword ptr [rdx - 512]{1to16}
// CHECK: encoding: [0x62,0xe5,0x7c,0xdf,0x6d,0x72,0x80]
          vcvttps2dqs zmm22 {k7} {z}, dword ptr [rdx - 512]{1to16}

// CHECK: vcvttps2qqs xmm22, xmm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x08,0x6d,0xf7]
          vcvttps2qqs xmm22, xmm23

// CHECK: vcvttps2qqs xmm22 {k7}, xmm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x0f,0x6d,0xf7]
          vcvttps2qqs xmm22 {k7}, xmm23

// CHECK: vcvttps2qqs xmm22 {k7} {z}, xmm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x8f,0x6d,0xf7]
          vcvttps2qqs xmm22 {k7} {z}, xmm23

// CHECK: vcvttps2qqs ymm22, xmm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x28,0x6d,0xf7]
          vcvttps2qqs ymm22, xmm23

// CHECK: vcvttps2qqs ymm22, xmm23, {sae}
// CHECK: encoding: [0x62,0xa5,0x79,0x18,0x6d,0xf7]
          vcvttps2qqs ymm22, xmm23, {sae}

// CHECK: vcvttps2qqs ymm22 {k7}, xmm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x2f,0x6d,0xf7]
          vcvttps2qqs ymm22 {k7}, xmm23

// CHECK: vcvttps2qqs ymm22 {k7} {z}, xmm23, {sae}
// CHECK: encoding: [0x62,0xa5,0x79,0x9f,0x6d,0xf7]
          vcvttps2qqs ymm22 {k7} {z}, xmm23, {sae}

// CHECK: vcvttps2qqs zmm22, ymm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x48,0x6d,0xf7]
          vcvttps2qqs zmm22, ymm23

// CHECK: vcvttps2qqs zmm22, ymm23, {sae}
// CHECK: encoding: [0x62,0xa5,0x7d,0x18,0x6d,0xf7]
          vcvttps2qqs zmm22, ymm23, {sae}

// CHECK: vcvttps2qqs zmm22 {k7}, ymm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x4f,0x6d,0xf7]
          vcvttps2qqs zmm22 {k7}, ymm23

// CHECK: vcvttps2qqs zmm22 {k7} {z}, ymm23, {sae}
// CHECK: encoding: [0x62,0xa5,0x7d,0x9f,0x6d,0xf7]
          vcvttps2qqs zmm22 {k7} {z}, ymm23, {sae}

// CHECK: vcvttps2qqs xmm22, qword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7d,0x08,0x6d,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvttps2qqs xmm22, qword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttps2qqs xmm22 {k7}, qword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7d,0x0f,0x6d,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvttps2qqs xmm22 {k7}, qword ptr [r8 + 4*rax + 291]

// CHECK: vcvttps2qqs xmm22, dword ptr [rip]{1to2}
// CHECK: encoding: [0x62,0xe5,0x7d,0x18,0x6d,0x35,0x00,0x00,0x00,0x00]
          vcvttps2qqs xmm22, dword ptr [rip]{1to2}

// CHECK: vcvttps2qqs xmm22, qword ptr [2*rbp - 256]
// CHECK: encoding: [0x62,0xe5,0x7d,0x08,0x6d,0x34,0x6d,0x00,0xff,0xff,0xff]
          vcvttps2qqs xmm22, qword ptr [2*rbp - 256]

// CHECK: vcvttps2qqs xmm22 {k7} {z}, qword ptr [rcx + 1016]
// CHECK: encoding: [0x62,0xe5,0x7d,0x8f,0x6d,0x71,0x7f]
          vcvttps2qqs xmm22 {k7} {z}, qword ptr [rcx + 1016]

// CHECK: vcvttps2qqs xmm22 {k7} {z}, dword ptr [rdx - 512]{1to2}
// CHECK: encoding: [0x62,0xe5,0x7d,0x9f,0x6d,0x72,0x80]
          vcvttps2qqs xmm22 {k7} {z}, dword ptr [rdx - 512]{1to2}

// CHECK: vcvttps2qqs ymm22, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7d,0x28,0x6d,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvttps2qqs ymm22, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttps2qqs ymm22 {k7}, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7d,0x2f,0x6d,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvttps2qqs ymm22 {k7}, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvttps2qqs ymm22, dword ptr [rip]{1to4}
// CHECK: encoding: [0x62,0xe5,0x7d,0x38,0x6d,0x35,0x00,0x00,0x00,0x00]
          vcvttps2qqs ymm22, dword ptr [rip]{1to4}

// CHECK: vcvttps2qqs ymm22, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe5,0x7d,0x28,0x6d,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vcvttps2qqs ymm22, xmmword ptr [2*rbp - 512]

// CHECK: vcvttps2qqs ymm22 {k7} {z}, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe5,0x7d,0xaf,0x6d,0x71,0x7f]
          vcvttps2qqs ymm22 {k7} {z}, xmmword ptr [rcx + 2032]

// CHECK: vcvttps2qqs ymm22 {k7} {z}, dword ptr [rdx - 512]{1to4}
// CHECK: encoding: [0x62,0xe5,0x7d,0xbf,0x6d,0x72,0x80]
          vcvttps2qqs ymm22 {k7} {z}, dword ptr [rdx - 512]{1to4}

// CHECK: vcvttps2qqs zmm22, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7d,0x48,0x6d,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvttps2qqs zmm22, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttps2qqs zmm22 {k7}, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7d,0x4f,0x6d,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvttps2qqs zmm22 {k7}, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vcvttps2qqs zmm22, dword ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe5,0x7d,0x58,0x6d,0x35,0x00,0x00,0x00,0x00]
          vcvttps2qqs zmm22, dword ptr [rip]{1to8}

// CHECK: vcvttps2qqs zmm22, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe5,0x7d,0x48,0x6d,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vcvttps2qqs zmm22, ymmword ptr [2*rbp - 1024]

// CHECK: vcvttps2qqs zmm22 {k7} {z}, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe5,0x7d,0xcf,0x6d,0x71,0x7f]
          vcvttps2qqs zmm22 {k7} {z}, ymmword ptr [rcx + 4064]

// CHECK: vcvttps2qqs zmm22 {k7} {z}, dword ptr [rdx - 512]{1to8}
// CHECK: encoding: [0x62,0xe5,0x7d,0xdf,0x6d,0x72,0x80]
          vcvttps2qqs zmm22 {k7} {z}, dword ptr [rdx - 512]{1to8}

// CHECK: vcvttps2udqs xmm22, xmm23
// CHECK: encoding: [0x62,0xa5,0x7c,0x08,0x6c,0xf7]
          vcvttps2udqs xmm22, xmm23

// CHECK: vcvttps2udqs xmm22 {k7}, xmm23
// CHECK: encoding: [0x62,0xa5,0x7c,0x0f,0x6c,0xf7]
          vcvttps2udqs xmm22 {k7}, xmm23

// CHECK: vcvttps2udqs xmm22 {k7} {z}, xmm23
// CHECK: encoding: [0x62,0xa5,0x7c,0x8f,0x6c,0xf7]
          vcvttps2udqs xmm22 {k7} {z}, xmm23

// CHECK: vcvttps2udqs ymm22, ymm23
// CHECK: encoding: [0x62,0xa5,0x7c,0x28,0x6c,0xf7]
          vcvttps2udqs ymm22, ymm23

// CHECK: vcvttps2udqs ymm22, ymm23, {sae}
// CHECK: encoding: [0x62,0xa5,0x78,0x18,0x6c,0xf7]
          vcvttps2udqs ymm22, ymm23, {sae}

// CHECK: vcvttps2udqs ymm22 {k7}, ymm23
// CHECK: encoding: [0x62,0xa5,0x7c,0x2f,0x6c,0xf7]
          vcvttps2udqs ymm22 {k7}, ymm23

// CHECK: vcvttps2udqs ymm22 {k7} {z}, ymm23, {sae}
// CHECK: encoding: [0x62,0xa5,0x78,0x9f,0x6c,0xf7]
          vcvttps2udqs ymm22 {k7} {z}, ymm23, {sae}

// CHECK: vcvttps2udqs zmm22, zmm23
// CHECK: encoding: [0x62,0xa5,0x7c,0x48,0x6c,0xf7]
          vcvttps2udqs zmm22, zmm23

// CHECK: vcvttps2udqs zmm22, zmm23, {sae}
// CHECK: encoding: [0x62,0xa5,0x7c,0x18,0x6c,0xf7]
          vcvttps2udqs zmm22, zmm23, {sae}

// CHECK: vcvttps2udqs zmm22 {k7}, zmm23
// CHECK: encoding: [0x62,0xa5,0x7c,0x4f,0x6c,0xf7]
          vcvttps2udqs zmm22 {k7}, zmm23

// CHECK: vcvttps2udqs zmm22 {k7} {z}, zmm23, {sae}
// CHECK: encoding: [0x62,0xa5,0x7c,0x9f,0x6c,0xf7]
          vcvttps2udqs zmm22 {k7} {z}, zmm23, {sae}

// CHECK: vcvttps2udqs xmm22, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7c,0x08,0x6c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvttps2udqs xmm22, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttps2udqs xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7c,0x0f,0x6c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvttps2udqs xmm22 {k7}, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvttps2udqs xmm22, dword ptr [rip]{1to4}
// CHECK: encoding: [0x62,0xe5,0x7c,0x18,0x6c,0x35,0x00,0x00,0x00,0x00]
          vcvttps2udqs xmm22, dword ptr [rip]{1to4}

// CHECK: vcvttps2udqs xmm22, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe5,0x7c,0x08,0x6c,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vcvttps2udqs xmm22, xmmword ptr [2*rbp - 512]

// CHECK: vcvttps2udqs xmm22 {k7} {z}, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe5,0x7c,0x8f,0x6c,0x71,0x7f]
          vcvttps2udqs xmm22 {k7} {z}, xmmword ptr [rcx + 2032]

// CHECK: vcvttps2udqs xmm22 {k7} {z}, dword ptr [rdx - 512]{1to4}
// CHECK: encoding: [0x62,0xe5,0x7c,0x9f,0x6c,0x72,0x80]
          vcvttps2udqs xmm22 {k7} {z}, dword ptr [rdx - 512]{1to4}

// CHECK: vcvttps2udqs ymm22, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7c,0x28,0x6c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvttps2udqs ymm22, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttps2udqs ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7c,0x2f,0x6c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvttps2udqs ymm22 {k7}, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vcvttps2udqs ymm22, dword ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe5,0x7c,0x38,0x6c,0x35,0x00,0x00,0x00,0x00]
          vcvttps2udqs ymm22, dword ptr [rip]{1to8}

// CHECK: vcvttps2udqs ymm22, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe5,0x7c,0x28,0x6c,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vcvttps2udqs ymm22, ymmword ptr [2*rbp - 1024]

// CHECK: vcvttps2udqs ymm22 {k7} {z}, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe5,0x7c,0xaf,0x6c,0x71,0x7f]
          vcvttps2udqs ymm22 {k7} {z}, ymmword ptr [rcx + 4064]

// CHECK: vcvttps2udqs ymm22 {k7} {z}, dword ptr [rdx - 512]{1to8}
// CHECK: encoding: [0x62,0xe5,0x7c,0xbf,0x6c,0x72,0x80]
          vcvttps2udqs ymm22 {k7} {z}, dword ptr [rdx - 512]{1to8}

// CHECK: vcvttps2udqs zmm22, zmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7c,0x48,0x6c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvttps2udqs zmm22, zmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttps2udqs zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7c,0x4f,0x6c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvttps2udqs zmm22 {k7}, zmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvttps2udqs zmm22, dword ptr [rip]{1to16}
// CHECK: encoding: [0x62,0xe5,0x7c,0x58,0x6c,0x35,0x00,0x00,0x00,0x00]
          vcvttps2udqs zmm22, dword ptr [rip]{1to16}

// CHECK: vcvttps2udqs zmm22, zmmword ptr [2*rbp - 2048]
// CHECK: encoding: [0x62,0xe5,0x7c,0x48,0x6c,0x34,0x6d,0x00,0xf8,0xff,0xff]
          vcvttps2udqs zmm22, zmmword ptr [2*rbp - 2048]

// CHECK: vcvttps2udqs zmm22 {k7} {z}, zmmword ptr [rcx + 8128]
// CHECK: encoding: [0x62,0xe5,0x7c,0xcf,0x6c,0x71,0x7f]
          vcvttps2udqs zmm22 {k7} {z}, zmmword ptr [rcx + 8128]

// CHECK: vcvttps2udqs zmm22 {k7} {z}, dword ptr [rdx - 512]{1to16}
// CHECK: encoding: [0x62,0xe5,0x7c,0xdf,0x6c,0x72,0x80]
          vcvttps2udqs zmm22 {k7} {z}, dword ptr [rdx - 512]{1to16}

// CHECK: vcvttps2uqqs xmm22, xmm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x08,0x6c,0xf7]
          vcvttps2uqqs xmm22, xmm23

// CHECK: vcvttps2uqqs xmm22 {k7}, xmm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x0f,0x6c,0xf7]
          vcvttps2uqqs xmm22 {k7}, xmm23

// CHECK: vcvttps2uqqs xmm22 {k7} {z}, xmm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x8f,0x6c,0xf7]
          vcvttps2uqqs xmm22 {k7} {z}, xmm23

// CHECK: vcvttps2uqqs ymm22, xmm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x28,0x6c,0xf7]
          vcvttps2uqqs ymm22, xmm23

// CHECK: vcvttps2uqqs ymm22, xmm23, {sae}
// CHECK: encoding: [0x62,0xa5,0x79,0x18,0x6c,0xf7]
          vcvttps2uqqs ymm22, xmm23, {sae}

// CHECK: vcvttps2uqqs ymm22 {k7}, xmm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x2f,0x6c,0xf7]
          vcvttps2uqqs ymm22 {k7}, xmm23

// CHECK: vcvttps2uqqs ymm22 {k7} {z}, xmm23, {sae}
// CHECK: encoding: [0x62,0xa5,0x79,0x9f,0x6c,0xf7]
          vcvttps2uqqs ymm22 {k7} {z}, xmm23, {sae}

// CHECK: vcvttps2uqqs zmm22, ymm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x48,0x6c,0xf7]
          vcvttps2uqqs zmm22, ymm23

// CHECK: vcvttps2uqqs zmm22, ymm23, {sae}
// CHECK: encoding: [0x62,0xa5,0x7d,0x18,0x6c,0xf7]
          vcvttps2uqqs zmm22, ymm23, {sae}

// CHECK: vcvttps2uqqs zmm22 {k7}, ymm23
// CHECK: encoding: [0x62,0xa5,0x7d,0x4f,0x6c,0xf7]
          vcvttps2uqqs zmm22 {k7}, ymm23

// CHECK: vcvttps2uqqs zmm22 {k7} {z}, ymm23, {sae}
// CHECK: encoding: [0x62,0xa5,0x7d,0x9f,0x6c,0xf7]
          vcvttps2uqqs zmm22 {k7} {z}, ymm23, {sae}

// CHECK: vcvttps2uqqs xmm22, qword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7d,0x08,0x6c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvttps2uqqs xmm22, qword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttps2uqqs xmm22 {k7}, qword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7d,0x0f,0x6c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvttps2uqqs xmm22 {k7}, qword ptr [r8 + 4*rax + 291]

// CHECK: vcvttps2uqqs xmm22, dword ptr [rip]{1to2}
// CHECK: encoding: [0x62,0xe5,0x7d,0x18,0x6c,0x35,0x00,0x00,0x00,0x00]
          vcvttps2uqqs xmm22, dword ptr [rip]{1to2}

// CHECK: vcvttps2uqqs xmm22, qword ptr [2*rbp - 256]
// CHECK: encoding: [0x62,0xe5,0x7d,0x08,0x6c,0x34,0x6d,0x00,0xff,0xff,0xff]
          vcvttps2uqqs xmm22, qword ptr [2*rbp - 256]

// CHECK: vcvttps2uqqs xmm22 {k7} {z}, qword ptr [rcx + 1016]
// CHECK: encoding: [0x62,0xe5,0x7d,0x8f,0x6c,0x71,0x7f]
          vcvttps2uqqs xmm22 {k7} {z}, qword ptr [rcx + 1016]

// CHECK: vcvttps2uqqs xmm22 {k7} {z}, dword ptr [rdx - 512]{1to2}
// CHECK: encoding: [0x62,0xe5,0x7d,0x9f,0x6c,0x72,0x80]
          vcvttps2uqqs xmm22 {k7} {z}, dword ptr [rdx - 512]{1to2}

// CHECK: vcvttps2uqqs ymm22, xmmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7d,0x28,0x6c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvttps2uqqs ymm22, xmmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttps2uqqs ymm22 {k7}, xmmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7d,0x2f,0x6c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvttps2uqqs ymm22 {k7}, xmmword ptr [r8 + 4*rax + 291]

// CHECK: vcvttps2uqqs ymm22, dword ptr [rip]{1to4}
// CHECK: encoding: [0x62,0xe5,0x7d,0x38,0x6c,0x35,0x00,0x00,0x00,0x00]
          vcvttps2uqqs ymm22, dword ptr [rip]{1to4}

// CHECK: vcvttps2uqqs ymm22, xmmword ptr [2*rbp - 512]
// CHECK: encoding: [0x62,0xe5,0x7d,0x28,0x6c,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vcvttps2uqqs ymm22, xmmword ptr [2*rbp - 512]

// CHECK: vcvttps2uqqs ymm22 {k7} {z}, xmmword ptr [rcx + 2032]
// CHECK: encoding: [0x62,0xe5,0x7d,0xaf,0x6c,0x71,0x7f]
          vcvttps2uqqs ymm22 {k7} {z}, xmmword ptr [rcx + 2032]

// CHECK: vcvttps2uqqs ymm22 {k7} {z}, dword ptr [rdx - 512]{1to4}
// CHECK: encoding: [0x62,0xe5,0x7d,0xbf,0x6c,0x72,0x80]
          vcvttps2uqqs ymm22 {k7} {z}, dword ptr [rdx - 512]{1to4}

// CHECK: vcvttps2uqqs zmm22, ymmword ptr [rbp + 8*r14 + 268435456]
// CHECK: encoding: [0x62,0xa5,0x7d,0x48,0x6c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvttps2uqqs zmm22, ymmword ptr [rbp + 8*r14 + 268435456]

// CHECK: vcvttps2uqqs zmm22 {k7}, ymmword ptr [r8 + 4*rax + 291]
// CHECK: encoding: [0x62,0xc5,0x7d,0x4f,0x6c,0xb4,0x80,0x23,0x01,0x00,0x00]
          vcvttps2uqqs zmm22 {k7}, ymmword ptr [r8 + 4*rax + 291]

// CHECK: vcvttps2uqqs zmm22, dword ptr [rip]{1to8}
// CHECK: encoding: [0x62,0xe5,0x7d,0x58,0x6c,0x35,0x00,0x00,0x00,0x00]
          vcvttps2uqqs zmm22, dword ptr [rip]{1to8}

// CHECK: vcvttps2uqqs zmm22, ymmword ptr [2*rbp - 1024]
// CHECK: encoding: [0x62,0xe5,0x7d,0x48,0x6c,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vcvttps2uqqs zmm22, ymmword ptr [2*rbp - 1024]

// CHECK: vcvttps2uqqs zmm22 {k7} {z}, ymmword ptr [rcx + 4064]
// CHECK: encoding: [0x62,0xe5,0x7d,0xcf,0x6c,0x71,0x7f]
          vcvttps2uqqs zmm22 {k7} {z}, ymmword ptr [rcx + 4064]

// CHECK: vcvttps2uqqs zmm22 {k7} {z}, dword ptr [rdx - 512]{1to8}
// CHECK: encoding: [0x62,0xe5,0x7d,0xdf,0x6c,0x72,0x80]
          vcvttps2uqqs zmm22 {k7} {z}, dword ptr [rdx - 512]{1to8}

