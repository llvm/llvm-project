/*===--------------------------------------------------------------------------
 *              ATMI (Asynchronous Task and Memory Interface)
 *
 * This file is distributed under the MIT License. See LICENSE.txt for details.
 *===------------------------------------------------------------------------*/
#ifndef SRC_RUNTIME_INCLUDE_QUEUE_H_
#define SRC_RUNTIME_INCLUDE_QUEUE_H_

#include "atmi.h"
#include "hsa.h"
class ATLQueue {
 public:
  explicit ATLQueue(hsa_queue_t *q, atmi_place_t p = ATMI_PLACE_ANY(0))
      : queue_(q), place_(p) {}
  hsa_queue_t *queue() const { return queue_; }
  atmi_place_t place() const { return place_; }

  hsa_status_t set_place(atmi_place_t place);

 protected:
  hsa_queue_t *queue_;
  atmi_place_t place_;
};

class ATLCPUQueue : public ATLQueue {
 public:
  explicit ATLCPUQueue(hsa_queue_t *q, atmi_place_t p = ATMI_PLACE_ANY_CPU(0))
      : ATLQueue(q, p) {}
  hsa_status_t set_place(atmi_place_t place);
};

class ATLGPUQueue : public ATLQueue {
 public:
  explicit ATLGPUQueue(hsa_queue_t *q, atmi_place_t p = ATMI_PLACE_ANY_GPU(0))
      : ATLQueue(q, p) {}
  hsa_status_t set_place(atmi_place_t place);
};

#endif  // SRC_RUNTIME_INCLUDE_QUEUE_H_
