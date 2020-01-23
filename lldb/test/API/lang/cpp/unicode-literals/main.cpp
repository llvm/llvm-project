int main() { int argc = 0; char **argv = (char **)0;

    auto cs16 = u"hello world ྒྙྐ";
	auto cs32 = U"hello world ྒྙྐ";
    char16_t *s16 = (char16_t *)u"ﺸﺵۻ";
    char32_t *s32 = (char32_t *)U"ЕЙРГЖО";
    s32 = nullptr; // Set break point at this line.
    s32 = (char32_t *)U"෴";
    s16 = (char16_t *)u"色ハ匂ヘト散リヌルヲ";
    return 0;
}
