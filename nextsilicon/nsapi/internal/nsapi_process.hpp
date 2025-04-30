#pragma once

#include <nsapi/nsapi.hpp>

struct NSAPICommandMultiprocessSet : public NSAPICommand {
public:
  uint32_t ProcessIndex;
  uint32_t TotalProcesses;

  explicit NSAPICommandMultiprocessSet(uint32_t proc_index,
                                       uint32_t total_procs)
      : NSAPICommand(NSAPI_MULTIPROCESS_SET), ProcessIndex(proc_index),
        TotalProcesses(total_procs) {}
};
