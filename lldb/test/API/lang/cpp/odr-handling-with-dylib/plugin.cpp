#include "plugin.h"
#include "service.h"

struct Proxy : public Service {
  State *proxyState;
};

Proxy *gProxyThis = 0;

extern "C" {
void plugin_init() { gProxyThis = new Proxy; }

void plugin_entry() {}
}
