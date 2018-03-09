#include "bytes.h"
#include "words.h"

int main() {
  bool result = false;
  if (loopBytes())
    result |= true;
  if (loopWords())
    result |= true;

  if (result)
    return 0;

  return result;
}
