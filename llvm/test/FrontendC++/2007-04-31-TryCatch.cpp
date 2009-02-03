// RUN: %llvmgxx -S %s -o /dev/null

#include <locale>

namespace std 
{
  codecvt<char, char, mbstate_t>::
  codecvt(size_t __refs)
  : __codecvt_abstract_base<char, char, mbstate_t>(__refs),
  _M_c_locale_codecvt(_S_get_c_locale())
  { }
}
