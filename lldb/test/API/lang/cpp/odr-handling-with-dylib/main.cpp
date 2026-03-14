#include "plugin.h"

#define HIDE_FROM_PLUGIN 1
#include "service.h"

int main() {
  exported();
  plugin_init();
  plugin_entry();
  return 0;
}
