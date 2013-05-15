//===-- main.c --------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


int main (int argc, char const *argv[])
{
    auto cs16 = u"hello world ྒྙྐ";
	auto cs32 = U"hello world ྒྙྐ";
    char16_t *s16 = (char16_t *)u"ﺸﺵۻ";
    char32_t *s32 = (char32_t *)U"ЕЙРГЖО";
    s32 = nullptr; // Set break point at this line.
    s32 = (char32_t *)U"෴";
    s16 = (char16_t *)u"色ハ匂ヘト散リヌルヲ";
    return 0;
}
