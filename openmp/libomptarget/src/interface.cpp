//===-------- interface.cpp - Target independent OpenMP target RTL --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//
//
// Implementation of the interface to be used by Clang during the codegen of a
// target region.
//
//===----------------------------------------------------------------------===//

#include <omptarget.h>

#include "device.h"
#include "private.h"
#include "rtl.h"

#include <cassert>

////////////////////////////////////////////////////////////////////////////////
/// adds a target shared library to the target execution image
EXTERN void __tgt_register_lib(__tgt_bin_desc *desc) {
  RTLs.RegisterLib(desc);
}

////////////////////////////////////////////////////////////////////////////////
/// unloads a target shared library
EXTERN void __tgt_unregister_lib(__tgt_bin_desc *desc) {
  RTLs.UnregisterLib(desc);
}

// Following datatypes and functions (tgt_oldmap_type, combined_entry_t,
// translate_map, cleanup_map) will be removed once the compiler starts using
// the new map types.

// Old map types
enum tgt_oldmap_type {
  OMP_TGT_OLDMAPTYPE_TO          = 0x001, // copy data from host to device
  OMP_TGT_OLDMAPTYPE_FROM        = 0x002, // copy data from device to host
  OMP_TGT_OLDMAPTYPE_ALWAYS      = 0x004, // copy regardless of the ref. count
  OMP_TGT_OLDMAPTYPE_DELETE      = 0x008, // force unmapping of data
  OMP_TGT_OLDMAPTYPE_MAP_PTR     = 0x010, // map pointer as well as pointee
  OMP_TGT_OLDMAPTYPE_FIRST_MAP   = 0x020, // first occurrence of mapped variable
  OMP_TGT_OLDMAPTYPE_RETURN_PTR  = 0x040, // return TgtBase addr of mapped data
  OMP_TGT_OLDMAPTYPE_PRIVATE_PTR = 0x080, // private variable - not mapped
  OMP_TGT_OLDMAPTYPE_PRIVATE_VAL = 0x100  // copy by value - not mapped
};

// Temporary functions for map translation and cleanup
struct combined_entry_t {
  int num_members; // number of members in combined entry
  void *base_addr; // base address of combined entry
  void *begin_addr; // begin address of combined entry
  void *end_addr; // size of combined entry
};

static void translate_map(int32_t arg_num, void **args_base, void **args,
    int64_t *arg_sizes, int64_t *arg_types, int32_t &new_arg_num,
    void **&new_args_base, void **&new_args, int64_t *&new_arg_sizes,
    int64_t *&new_arg_types, bool is_target_construct) {
  if (arg_num <= 0) {
    DP("Nothing to translate\n");
    new_arg_num = 0;
    return;
  }

  // array of combined entries
  combined_entry_t *cmb_entries =
      (combined_entry_t *) alloca(arg_num * sizeof(combined_entry_t));
  // number of combined entries
  long num_combined = 0;
  // old entry is MAP_PTR?
  bool *is_ptr_old = (bool *) alloca(arg_num * sizeof(bool));
  // old entry is member of member_of[old] cmb_entry
  int *member_of = (int *) alloca(arg_num * sizeof(int));
  // temporary storage for modifications of the original arg_types
  int64_t *mod_arg_types = (int64_t *) alloca(arg_num  *sizeof(int64_t));

  DP("Translating %d map entries\n", arg_num);
  for (int i = 0; i < arg_num; ++i) {
    member_of[i] = -1;
    is_ptr_old[i] = false;
    mod_arg_types[i] = arg_types[i];
    // Scan previous entries to see whether this entry shares the same base
    for (int j = 0; j < i; ++j) {
      void *new_begin_addr = NULL;
      void *new_end_addr = NULL;

      if (mod_arg_types[i] & OMP_TGT_OLDMAPTYPE_MAP_PTR) {
        if (args_base[i] == args[j]) {
          if (!(mod_arg_types[j] & OMP_TGT_OLDMAPTYPE_MAP_PTR)) {
            DP("Entry %d has the same base as entry %d's begin address\n", i,
                j);
            new_begin_addr = args_base[i];
            new_end_addr = (char *)args_base[i] + sizeof(void *);
            assert(arg_sizes[j] == sizeof(void *));
            is_ptr_old[j] = true;
          } else {
            DP("Entry %d has the same base as entry %d's begin address, but "
                "%d's base was a MAP_PTR too\n", i, j, j);
            int32_t to_from_always_delete =
                OMP_TGT_OLDMAPTYPE_TO | OMP_TGT_OLDMAPTYPE_FROM |
                OMP_TGT_OLDMAPTYPE_ALWAYS | OMP_TGT_OLDMAPTYPE_DELETE;
            if (mod_arg_types[j] & to_from_always_delete) {
              DP("Resetting to/from/always/delete flags for entry %d because "
                  "it is only a pointer to pointer\n", j);
              mod_arg_types[j] &= ~to_from_always_delete;
            }
          }
        }
      } else {
        if (!(mod_arg_types[i] & OMP_TGT_OLDMAPTYPE_FIRST_MAP) &&
            args_base[i] == args_base[j]) {
          DP("Entry %d has the same base address as entry %d\n", i, j);
          new_begin_addr = args[i];
          new_end_addr = (char *)args[i] + arg_sizes[i];
        }
      }

      // If we have combined the entry with a previous one
      if (new_begin_addr) {
        int id;
        if(member_of[j] == -1) {
          // We have a new entry
          id = num_combined++;
          DP("Creating new combined entry %d for old entry %d\n", id, j);
          // Initialize new entry
          cmb_entries[id].num_members = 1;
          cmb_entries[id].base_addr = args_base[j];
          if (mod_arg_types[j] & OMP_TGT_OLDMAPTYPE_MAP_PTR) {
            cmb_entries[id].begin_addr = args_base[j];
            cmb_entries[id].end_addr = (char *)args_base[j] + arg_sizes[j];
          } else {
            cmb_entries[id].begin_addr = args[j];
            cmb_entries[id].end_addr = (char *)args[j] + arg_sizes[j];
          }
          member_of[j] = id;
        } else {
          // Reuse existing combined entry
          DP("Reusing existing combined entry %d\n", member_of[j]);
          id = member_of[j];
        }

        // Update combined entry
        DP("Adding entry %d to combined entry %d\n", i, id);
        cmb_entries[id].num_members++;
        // base_addr stays the same
        cmb_entries[id].begin_addr =
            std::min(cmb_entries[id].begin_addr, new_begin_addr);
        cmb_entries[id].end_addr =
            std::max(cmb_entries[id].end_addr, new_end_addr);
        member_of[i] = id;
        break;
      }
    }
  }

  DP("New entries: %ld combined + %d original\n", num_combined, arg_num);
  new_arg_num = arg_num + num_combined;
  new_args_base = (void **) malloc(new_arg_num * sizeof(void *));
  new_args = (void **) malloc(new_arg_num * sizeof(void *));
  new_arg_sizes = (int64_t *) malloc(new_arg_num * sizeof(int64_t));
  new_arg_types = (int64_t *) malloc(new_arg_num * sizeof(int64_t));

  const int64_t alignment = 8;

  int next_id = 0; // next ID
  int next_cid = 0; // next combined ID
  int *combined_to_new_id = (int *) alloca(num_combined * sizeof(int));
  for (int i = 0; i < arg_num; ++i) {
    // It is member_of
    if (member_of[i] == next_cid) {
      int cid = next_cid++; // ID of this combined entry
      int nid = next_id++; // ID of the new (global) entry
      combined_to_new_id[cid] = nid;
      DP("Combined entry %3d will become new entry %3d\n", cid, nid);

      int64_t padding = (int64_t)cmb_entries[cid].begin_addr % alignment;
      if (padding) {
        DP("Using a padding of %" PRId64 " for begin address " DPxMOD "\n",
            padding, DPxPTR(cmb_entries[cid].begin_addr));
        cmb_entries[cid].begin_addr =
            (char *)cmb_entries[cid].begin_addr - padding;
      }

      new_args_base[nid] = cmb_entries[cid].base_addr;
      new_args[nid] = cmb_entries[cid].begin_addr;
      new_arg_sizes[nid] = (int64_t) ((char *)cmb_entries[cid].end_addr -
          (char *)cmb_entries[cid].begin_addr);
      new_arg_types[nid] = OMP_TGT_MAPTYPE_TARGET_PARAM;
      DP("Entry %3d: base_addr " DPxMOD ", begin_addr " DPxMOD ", "
          "size %" PRId64 ", type 0x%" PRIx64 "\n", nid,
          DPxPTR(new_args_base[nid]), DPxPTR(new_args[nid]), new_arg_sizes[nid],
          new_arg_types[nid]);
    } else if (member_of[i] != -1) {
      DP("Combined entry %3d has been encountered before, do nothing\n",
          member_of[i]);
    }

    // Now that the combined entry (the one the old entry was a member of) has
    // been inserted into the new arguments list, proceed with the old entry.
    int nid = next_id++;
    DP("Old entry %3d will become new entry %3d\n", i, nid);

    new_args_base[nid] = args_base[i];
    new_args[nid] = args[i];
    new_arg_sizes[nid] = arg_sizes[i];
    int64_t old_type = mod_arg_types[i];

    if (is_ptr_old[i]) {
      // Reset TO and FROM flags
      old_type &= ~(OMP_TGT_OLDMAPTYPE_TO | OMP_TGT_OLDMAPTYPE_FROM);
    }

    if (member_of[i] == -1) {
      if (!is_target_construct)
        old_type &= ~OMP_TGT_MAPTYPE_TARGET_PARAM;
      new_arg_types[nid] = old_type;
      DP("Entry %3d: base_addr " DPxMOD ", begin_addr " DPxMOD ", size %" PRId64
          ", type 0x%" PRIx64 " (old entry %d not MEMBER_OF)\n", nid,
          DPxPTR(new_args_base[nid]), DPxPTR(new_args[nid]), new_arg_sizes[nid],
          new_arg_types[nid], i);
    } else {
      // Old entry is not FIRST_MAP
      old_type &= ~OMP_TGT_OLDMAPTYPE_FIRST_MAP;
      // Add MEMBER_OF
      int new_member_of = combined_to_new_id[member_of[i]];
      old_type |= ((int64_t)new_member_of + 1) << 48;
      new_arg_types[nid] = old_type;
      DP("Entry %3d: base_addr " DPxMOD ", begin_addr " DPxMOD ", size %" PRId64
        ", type 0x%" PRIx64 " (old entry %d MEMBER_OF %d)\n", nid,
        DPxPTR(new_args_base[nid]), DPxPTR(new_args[nid]), new_arg_sizes[nid],
        new_arg_types[nid], i, new_member_of);
    }
  }
}

static void cleanup_map(int32_t new_arg_num, void **new_args_base,
    void **new_args, int64_t *new_arg_sizes, int64_t *new_arg_types,
    int32_t arg_num, void **args_base) {
  if (new_arg_num > 0) {
    int offset = new_arg_num - arg_num;
    for (int32_t i = 0; i < arg_num; ++i) {
      // Restore old base address
      args_base[i] = new_args_base[i+offset];
    }
    free(new_args_base);
    free(new_args);
    free(new_arg_sizes);
    free(new_arg_types);
  }
}

/// creates host-to-target data mapping, stores it in the
/// libomptarget.so internal structure (an entry in a stack of data maps)
/// and passes the data to the device.
EXTERN void __tgt_target_data_begin(int64_t device_id, int32_t arg_num,
    void **args_base, void **args, int64_t *arg_sizes, int64_t *arg_types) {
  DP("Entering data begin region for device %ld with %d mappings\n", device_id,
     arg_num);

  // No devices available?
  if (device_id == OFFLOAD_DEVICE_DEFAULT) {
    device_id = omp_get_default_device();
    DP("Use default device id %ld\n", device_id);
  }

  if (CheckDeviceAndCtors(device_id) != OFFLOAD_SUCCESS) {
    DP("Failed to get device %ld ready\n", device_id);
    return;
  }

  DeviceTy& Device = Devices[device_id];

  // Translate maps
  int32_t new_arg_num;
  void **new_args_base;
  void **new_args;
  int64_t *new_arg_sizes;
  int64_t *new_arg_types;
  translate_map(arg_num, args_base, args, arg_sizes, arg_types, new_arg_num,
      new_args_base, new_args, new_arg_sizes, new_arg_types, false);

  //target_data_begin(Device, arg_num, args_base, args, arg_sizes, arg_types);
  target_data_begin(Device, new_arg_num, new_args_base, new_args, new_arg_sizes,
      new_arg_types);

  // Cleanup translation memory
  cleanup_map(new_arg_num, new_args_base, new_args, new_arg_sizes,
      new_arg_types, arg_num, args_base);
}

EXTERN void __tgt_target_data_begin_nowait(int64_t device_id, int32_t arg_num,
    void **args_base, void **args, int64_t *arg_sizes, int64_t *arg_types,
    int32_t depNum, void *depList, int32_t noAliasDepNum,
    void *noAliasDepList) {
  if (depNum + noAliasDepNum > 0)
    __kmpc_omp_taskwait(NULL, 0);

  __tgt_target_data_begin(device_id, arg_num, args_base, args, arg_sizes,
                          arg_types);
}

/// passes data from the target, releases target memory and destroys
/// the host-target mapping (top entry from the stack of data maps)
/// created by the last __tgt_target_data_begin.
EXTERN void __tgt_target_data_end(int64_t device_id, int32_t arg_num,
    void **args_base, void **args, int64_t *arg_sizes, int64_t *arg_types) {
  DP("Entering data end region with %d mappings\n", arg_num);

  // No devices available?
  if (device_id == OFFLOAD_DEVICE_DEFAULT) {
    device_id = omp_get_default_device();
  }

  RTLsMtx.lock();
  size_t Devices_size = Devices.size();
  RTLsMtx.unlock();
  if (Devices_size <= (size_t)device_id) {
    DP("Device ID  %ld does not have a matching RTL.\n", device_id);
    return;
  }

  DeviceTy &Device = Devices[device_id];
  if (!Device.IsInit) {
    DP("uninit device: ignore");
    return;
  }

  // Translate maps
  int32_t new_arg_num;
  void **new_args_base;
  void **new_args;
  int64_t *new_arg_sizes;
  int64_t *new_arg_types;
  translate_map(arg_num, args_base, args, arg_sizes, arg_types, new_arg_num,
      new_args_base, new_args, new_arg_sizes, new_arg_types, false);

  //target_data_end(Device, arg_num, args_base, args, arg_sizes, arg_types);
  target_data_end(Device, new_arg_num, new_args_base, new_args, new_arg_sizes,
      new_arg_types);

  // Cleanup translation memory
  cleanup_map(new_arg_num, new_args_base, new_args, new_arg_sizes,
      new_arg_types, arg_num, args_base);
}

EXTERN void __tgt_target_data_end_nowait(int64_t device_id, int32_t arg_num,
    void **args_base, void **args, int64_t *arg_sizes, int64_t *arg_types,
    int32_t depNum, void *depList, int32_t noAliasDepNum,
    void *noAliasDepList) {
  if (depNum + noAliasDepNum > 0)
    __kmpc_omp_taskwait(NULL, 0);

  __tgt_target_data_end(device_id, arg_num, args_base, args, arg_sizes,
                        arg_types);
}

EXTERN void __tgt_target_data_update(int64_t device_id, int32_t arg_num,
    void **args_base, void **args, int64_t *arg_sizes, int64_t *arg_types) {
  DP("Entering data update with %d mappings\n", arg_num);

  // No devices available?
  if (device_id == OFFLOAD_DEVICE_DEFAULT) {
    device_id = omp_get_default_device();
  }

  if (CheckDeviceAndCtors(device_id) != OFFLOAD_SUCCESS) {
    DP("Failed to get device %ld ready\n", device_id);
    return;
  }

  DeviceTy& Device = Devices[device_id];
  target_data_update(Device, arg_num, args_base, args, arg_sizes, arg_types);
}

EXTERN void __tgt_target_data_update_nowait(
    int64_t device_id, int32_t arg_num, void **args_base, void **args,
    int64_t *arg_sizes, int64_t *arg_types, int32_t depNum, void *depList,
    int32_t noAliasDepNum, void *noAliasDepList) {
  if (depNum + noAliasDepNum > 0)
    __kmpc_omp_taskwait(NULL, 0);

  __tgt_target_data_update(device_id, arg_num, args_base, args, arg_sizes,
                           arg_types);
}

EXTERN int __tgt_target(int64_t device_id, void *host_ptr, int32_t arg_num,
    void **args_base, void **args, int64_t *arg_sizes, int64_t *arg_types) {
  DP("Entering target region with entry point " DPxMOD " and device Id %ld\n",
     DPxPTR(host_ptr), device_id);

  if (device_id == OFFLOAD_DEVICE_DEFAULT) {
    device_id = omp_get_default_device();
  }

  if (CheckDeviceAndCtors(device_id) != OFFLOAD_SUCCESS) {
    DP("Failed to get device %ld ready\n", device_id);
    return OFFLOAD_FAIL;
  }

  // Translate maps
  int32_t new_arg_num;
  void **new_args_base;
  void **new_args;
  int64_t *new_arg_sizes;
  int64_t *new_arg_types;
  translate_map(arg_num, args_base, args, arg_sizes, arg_types, new_arg_num,
      new_args_base, new_args, new_arg_sizes, new_arg_types, true);

  //return target(device_id, host_ptr, arg_num, args_base, args, arg_sizes,
  //    arg_types, 0, 0, false /*team*/, false /*recursive*/);
  int rc = target(device_id, host_ptr, new_arg_num, new_args_base, new_args,
      new_arg_sizes, new_arg_types, 0, 0, false /*team*/);

  // Cleanup translation memory
  cleanup_map(new_arg_num, new_args_base, new_args, new_arg_sizes,
      new_arg_types, arg_num, args_base);

  return rc;
}

EXTERN int __tgt_target_nowait(int64_t device_id, void *host_ptr,
    int32_t arg_num, void **args_base, void **args, int64_t *arg_sizes,
    int64_t *arg_types, int32_t depNum, void *depList, int32_t noAliasDepNum,
    void *noAliasDepList) {
  if (depNum + noAliasDepNum > 0)
    __kmpc_omp_taskwait(NULL, 0);

  return __tgt_target(device_id, host_ptr, arg_num, args_base, args, arg_sizes,
                      arg_types);
}

EXTERN int __tgt_target_teams(int64_t device_id, void *host_ptr,
    int32_t arg_num, void **args_base, void **args, int64_t *arg_sizes,
    int64_t *arg_types, int32_t team_num, int32_t thread_limit) {
  DP("Entering target region with entry point " DPxMOD " and device Id %ld\n",
     DPxPTR(host_ptr), device_id);

  if (device_id == OFFLOAD_DEVICE_DEFAULT) {
    device_id = omp_get_default_device();
  }

  if (CheckDeviceAndCtors(device_id) != OFFLOAD_SUCCESS) {
    DP("Failed to get device %ld ready\n", device_id);
    return OFFLOAD_FAIL;
  }

  // Translate maps
  int32_t new_arg_num;
  void **new_args_base;
  void **new_args;
  int64_t *new_arg_sizes;
  int64_t *new_arg_types;
  translate_map(arg_num, args_base, args, arg_sizes, arg_types, new_arg_num,
      new_args_base, new_args, new_arg_sizes, new_arg_types, true);

  //return target(device_id, host_ptr, arg_num, args_base, args, arg_sizes,
  //              arg_types, team_num, thread_limit, true /*team*/,
  //              false /*recursive*/);
  int rc = target(device_id, host_ptr, new_arg_num, new_args_base, new_args,
      new_arg_sizes, new_arg_types, team_num, thread_limit, true /*team*/);

  // Cleanup translation memory
  cleanup_map(new_arg_num, new_args_base, new_args, new_arg_sizes,
      new_arg_types, arg_num, args_base);

  return rc;
}

EXTERN int __tgt_target_teams_nowait(int64_t device_id, void *host_ptr,
    int32_t arg_num, void **args_base, void **args, int64_t *arg_sizes,
    int64_t *arg_types, int32_t team_num, int32_t thread_limit, int32_t depNum,
    void *depList, int32_t noAliasDepNum, void *noAliasDepList) {
  if (depNum + noAliasDepNum > 0)
    __kmpc_omp_taskwait(NULL, 0);

  return __tgt_target_teams(device_id, host_ptr, arg_num, args_base, args,
                            arg_sizes, arg_types, team_num, thread_limit);
}


// The trip count mechanism will be revised - this scheme is not thread-safe.
EXTERN void __kmpc_push_target_tripcount(int64_t device_id,
    uint64_t loop_tripcount) {
  if (device_id == OFFLOAD_DEVICE_DEFAULT) {
    device_id = omp_get_default_device();
  }

  if (CheckDeviceAndCtors(device_id) != OFFLOAD_SUCCESS) {
    DP("Failed to get device %ld ready\n", device_id);
    return;
  }

  DP("__kmpc_push_target_tripcount(%ld, %" PRIu64 ")\n", device_id,
      loop_tripcount);
  Devices[device_id].loopTripCnt = loop_tripcount;
}
