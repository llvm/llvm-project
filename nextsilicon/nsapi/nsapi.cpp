#include "nsapi.hpp"

#include <cerrno>
#include <memory>

extern "C" {

NSAPIHandler *nsapi_get_current_handler();

void nsapi_register_current_handler(NSAPIHandler *handler);
}
class NSAPIDefaultHandler : public NSAPIHandler {
private:
  struct private_ctor_tag {
    explicit private_ctor_tag() = default;
  };

public:
  explicit NSAPIDefaultHandler([[maybe_unused]] private_ctor_tag tag) {}

  ~NSAPIDefaultHandler() = default;

  NSAPIDefaultHandler(const NSAPIDefaultHandler &) = delete;
  NSAPIDefaultHandler(NSAPIDefaultHandler &&) = delete;
  NSAPIDefaultHandler &operator=(const NSAPIDefaultHandler &) = delete;
  NSAPIDefaultHandler &operator=(NSAPIDefaultHandler &&) = delete;

  static NSAPIDefaultHandler &instance() {
    static NSAPIDefaultHandler g_handler{private_ctor_tag{}};
    return g_handler;
  }

  // By default just return error
  virtual int Execute([[maybe_unused]] NSAPICommand &cmd) { return -ENOTSUP; }
};

// Set default handler (if there's no NextSilicon backend)
NSAPIHandler *NSAPIHandler::_current = &NSAPIDefaultHandler::instance();

NSAPIHandler *nsapi_get_current_handler() { return &NSAPIHandler::Current(); }

void nsapi_register_current_handler(NSAPIHandler *handler) {
  NSAPIHandler::Register(handler);
}
