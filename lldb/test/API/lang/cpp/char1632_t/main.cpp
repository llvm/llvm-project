#include <assert.h>
#include <string>

#define UASZ 64

template<class T, int N>
void copy_char_seq (T (&arr)[N], const T* src)
{
    size_t src_len = std::char_traits<T>::length(src);
    assert(src_len < N);

    std::char_traits<T>::copy(arr, src, src_len);
    arr[src_len] = 0;
}

int main (int argc, char const *argv[])
{
    char16_t as16[UASZ];
    char32_t as32[UASZ];
    auto cs16_zero = (char16_t)0;
    auto cs32_zero = (char32_t)0;
    auto cs16 = u"hello world ྒྙྐ";
    auto cs32 = U"hello world ྒྙྐ";
    char16_t *s16 = (char16_t *)u"ﺸﺵۻ";
    char32_t *s32 = (char32_t *)U"ЕЙРГЖО";
    copy_char_seq(as16, s16);
    copy_char_seq(as32, s32);
    s32 = nullptr; // breakpoint1
    s32 = (char32_t *)U"෴";
    s16 = (char16_t *)u"色ハ匂ヘト散リヌルヲ";
    copy_char_seq(as16, s16);
    copy_char_seq(as32, s32);
    s32 = nullptr; // breakpoint2
    return 0;
}
