/*===--------------------------------------------------------------------------
 *              ATMI (Asynchronous Task and Memory Interface)
 *
 * This file is distributed under the MIT License. See LICENSE.txt for details.
 *===------------------------------------------------------------------------*/
extern ATLMachine g_atl_machine;
template <typename T>
T& get_processor(atmi_place_t place) {
  int dev_id = place.device_id;
  if(dev_id == -1) {
    // user is asking runtime to pick a device
    // TODO(ashwinma): best device of this type? pick 0 for now
    dev_id = 0;
  }
  return g_atl_machine.processors<T>()[dev_id];
}
