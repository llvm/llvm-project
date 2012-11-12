#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/ManagedStatic.h"
#include <cassert>
#include <xlocale.h>


namespace {
  struct locale_holder {
    locale_holder()
    : l(newlocale(LC_CTYPE_MASK,"en_US.UTF-8",LC_GLOBAL_LOCALE))
    {
      assert(NULL!=l);
    }
    ~locale_holder() {
      freelocale(l);
    }

    int mbswidth(llvm::SmallString<16> s) const {
       // this implementation assumes no '\0' in s
      assert(s.size()==strlen(s.c_str()));

      size_t size = mbstowcs_l(NULL,s.c_str(),0,l);
      assert(size!=(size_t)-1);
      if (size==0)
        return 0;
      llvm::SmallVector<wchar_t,200> ws(size);
      size = mbstowcs_l(&ws[0],s.c_str(),ws.size(),l);
      assert(ws.size()==size);
      return wcswidth_l(&ws[0],ws.size(),l);
    }

    int isprint(int c) const {
      return iswprint_l(c,l);
    }

  private:

    locale_t l;
  };

  llvm::ManagedStatic<locale_holder> l;
}

namespace llvm {
namespace sys {
namespace locale {

int columnWidth(StringRef s) {
  int width = l->mbswidth(s);
  assert(width>=0);
  return width;
}

bool isPrint(int c) {
  return l->isprint(c);
}

}
}
}
