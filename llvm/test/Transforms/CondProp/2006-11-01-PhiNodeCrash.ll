; RUN: llvm-as < %s | opt -condprop -disable-output
; PR979

target datalayout = "e-p:32:32"
target triple = "i686-pc-linux-gnu"
deplibs = [ "c", "crtend" ]
	%struct.IO_APIC_reg_00 = type { i32 }
	%struct.Qdisc = type { i32 (%struct.sk_buff*, %struct.Qdisc*)*, %struct.sk_buff* (%struct.Qdisc*)*, i32, %struct.Qdisc_ops*, %struct.Qdisc*, i32, %struct.IO_APIC_reg_00, %struct.sk_buff_head, %struct.net_device*, %struct.tc_stats, i32 (%struct.sk_buff*, %struct.Qdisc*)*, %struct.Qdisc*, [1 x i8] }
	%struct.Qdisc_class_ops = type { i32 (%struct.Qdisc*, i32, %struct.Qdisc*, %struct.Qdisc**)*, %struct.Qdisc* (%struct.Qdisc*, i32)*, i32 (%struct.Qdisc*, i32)*, void (%struct.Qdisc*, i32)*, i32 (%struct.Qdisc*, i32, i32, %struct._agp_version**, i32*)*, i32 (%struct.Qdisc*, i32)*, void (%struct.Qdisc*, %struct.qdisc_walker*)*, %struct.tcf_proto** (%struct.Qdisc*, i32)*, i32 (%struct.Qdisc*, i32, i32)*, void (%struct.Qdisc*, i32)*, i32 (%struct.Qdisc*, i32, %struct.sk_buff*, %struct.tcmsg*)* }
	%struct.Qdisc_ops = type { %struct.Qdisc_ops*, %struct.Qdisc_class_ops*, [16 x i8], i32, i32 (%struct.sk_buff*, %struct.Qdisc*)*, %struct.sk_buff* (%struct.Qdisc*)*, i32 (%struct.sk_buff*, %struct.Qdisc*)*, i32 (%struct.Qdisc*)*, i32 (%struct.Qdisc*, %struct._agp_version*)*, void (%struct.Qdisc*)*, void (%struct.Qdisc*)*, i32 (%struct.Qdisc*, %struct._agp_version*)*, i32 (%struct.Qdisc*, %struct.sk_buff*)* }
	%struct.ViceFid = type { i32, i32, i32 }
	%struct.__wait_queue = type { i32, %struct.task_struct*, %struct.list_head }
	%struct.__wait_queue_head = type { %struct.IO_APIC_reg_00, %struct.list_head }
	%struct._agp_version = type { i16, i16 }
	%struct._drm_i810_overlay_t = type { i32, i32 }
	%struct.address_space = type { %struct.list_head, %struct.list_head, %struct.list_head, i32, %struct.address_space_operations*, %struct.inode*, %struct.vm_area_struct*, %struct.vm_area_struct*, %struct.IO_APIC_reg_00, i32 }
	%struct.address_space_operations = type { i32 (%struct.page*)*, i32 (%struct.file*, %struct.page*)*, i32 (%struct.page*)*, i32 (%struct.file*, %struct.page*, i32, i32)*, i32 (%struct.file*, %struct.page*, i32, i32)*, i32 (%struct.address_space*, i32)*, i32 (%struct.page*, i32)*, i32 (%struct.page*, i32)*, i32 (i32, %struct.inode*, %struct.kiobuf*, i32, i32)*, i32 (i32, %struct.file*, %struct.kiobuf*, i32, i32)*, void (%struct.page*)* }
	%struct.audio_buf_info = type { i32, i32, i32, i32 }
	%struct.autofs_packet_hdr = type { i32, i32 }
	%struct.block_device = type { %struct.list_head, %struct.IO_APIC_reg_00, %struct.inode*, i16, i32, %struct.block_device_operations*, %struct.semaphore, %struct.list_head }
	%struct.block_device_operations = type { i32 (%struct.inode*, %struct.file*)*, i32 (%struct.inode*, %struct.file*)*, i32 (%struct.inode*, %struct.file*, i32, i32)*, i32 (i16)*, i32 (i16)*, %struct.module* }
	%struct.bluez_skb_cb = type { i32 }
	%struct.buffer_head = type { %struct.buffer_head*, i32, i16, i16, i16, %struct.IO_APIC_reg_00, i16, i32, i32, %struct.buffer_head*, %struct.buffer_head*, %struct.buffer_head*, %struct.buffer_head*, %struct.buffer_head**, i8*, %struct.page*, void (%struct.buffer_head*, i32)*, i8*, i32, %struct.__wait_queue_head, %struct.list_head }
	%struct.char_device = type { %struct.list_head, %struct.IO_APIC_reg_00, i16, %struct.IO_APIC_reg_00, %struct.semaphore }
	%struct.completion = type { i32, %struct.__wait_queue_head }
	%struct.cramfs_info = type { i32, i32, i32, i32 }
	%struct.dentry = type { %struct.IO_APIC_reg_00, i32, %struct.inode*, %struct.dentry*, %struct.list_head, %struct.list_head, %struct.list_head, %struct.list_head, %struct.list_head, i32, %struct.qstr, i32, %struct.dentry_operations*, %struct.super_block*, i32, i8*, [16 x i8] }
	%struct.dentry_operations = type { i32 (%struct.dentry*, i32)*, i32 (%struct.dentry*, %struct.qstr*)*, i32 (%struct.dentry*, %struct.qstr*, %struct.qstr*)*, i32 (%struct.dentry*)*, void (%struct.dentry*)*, void (%struct.dentry*, %struct.inode*)* }
	%struct.dev_mc_list = type { %struct.dev_mc_list*, [8 x i8], i8, i32, i32 }
	%struct.dnotify_struct = type { %struct.dnotify_struct*, i32, i32, %struct.file*, %struct.files_struct* }
	%struct.dquot = type { %struct.list_head, %struct.list_head, %struct.list_head, %struct.__wait_queue_head, %struct.__wait_queue_head, i32, i32, %struct.super_block*, i32, i16, i64, i16, i16, i32, %struct.mem_dqblk }
	%struct.dquot_operations = type { void (%struct.inode*, i32)*, void (%struct.inode*)*, i32 (%struct.inode*, i64, i32)*, i32 (%struct.inode*, i32)*, void (%struct.inode*, i64)*, void (%struct.inode*, i32)*, i32 (%struct.inode*, %struct.iattr*)*, i32 (%struct.dquot*)* }
	%struct.drm_clip_rect = type { i16, i16, i16, i16 }
	%struct.drm_ctx_priv_map = type { i32, i8* }
	%struct.drm_mga_indices = type { i32, i32, i32, i32 }
	%struct.dst_entry = type { %struct.dst_entry*, %struct.IO_APIC_reg_00, i32, %struct.net_device*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %struct.neighbour*, %struct.hh_cache*, i32 (%struct.sk_buff*)*, i32 (%struct.sk_buff*)*, %struct.dst_ops*, [0 x i8] }
	%struct.dst_ops = type { i16, i16, i32, i32 ()*, %struct.dst_entry* (%struct.dst_entry*, i32)*, %struct.dst_entry* (%struct.dst_entry*, %struct.sk_buff*)*, void (%struct.dst_entry*)*, %struct.dst_entry* (%struct.dst_entry*)*, void (%struct.sk_buff*)*, i32, %struct.IO_APIC_reg_00, %struct.kmem_cache_s* }
	%struct.e820entry = type { i64, i64, i32 }
	%struct.exec_domain = type { i8*, void (i32, %struct.pt_regs*)*, i8, i8, i32*, i32*, %struct.map_segment*, %struct.map_segment*, %struct.map_segment*, %struct.map_segment*, %struct.module*, %struct.exec_domain* }
	%struct.ext2_inode_info = type { [15 x i32], i32, i32, i8, i8, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
	%struct.ext3_inode_info = type { [15 x i32], i32, i32, i32, i32, i32, i32, i32, i32, i32, %struct.list_head, i64, %struct.rw_semaphore }
	%struct.fasync_struct = type { i32, i32, %struct.fasync_struct*, %struct.file* }
	%struct.file = type { %struct.list_head, %struct.dentry*, %struct.vfsmount*, %struct.file_operations*, %struct.IO_APIC_reg_00, i32, i16, i64, i32, i32, i32, i32, i32, %struct.audio_buf_info, i32, i32, i32, i32, i8*, %struct.kiobuf*, i32 }
	%struct.file_lock = type { %struct.file_lock*, %struct.list_head, %struct.list_head, %struct.files_struct*, i32, %struct.__wait_queue_head, %struct.file*, i8, i8, i64, i64, void (%struct.file_lock*)*, void (%struct.file_lock*)*, void (%struct.file_lock*)*, %struct.fasync_struct*, i32, { %struct.nfs_lock_info } }
	%struct.file_operations = type { %struct.module*, i64 (%struct.file*, i64, i32)*, i32 (%struct.file*, i8*, i32, i64*)*, i32 (%struct.file*, i8*, i32, i64*)*, i32 (%struct.file*, i8*, i32 (i8*, i8*, i32, i64, i32, i32)*)*, i32 (%struct.file*, %struct.poll_table_struct*)*, i32 (%struct.inode*, %struct.file*, i32, i32)*, i32 (%struct.file*, %struct.vm_area_struct*)*, i32 (%struct.inode*, %struct.file*)*, i32 (%struct.file*)*, i32 (%struct.inode*, %struct.file*)*, i32 (%struct.file*, %struct.dentry*, i32)*, i32 (i32, %struct.file*, i32)*, i32 (%struct.file*, i32, %struct.file_lock*)*, i32 (%struct.file*, %struct.iovec*, i32, i64*)*, i32 (%struct.file*, %struct.iovec*, i32, i64*)*, i32 (%struct.file*, %struct.page*, i32, i32, i64*, i32)*, i32 (%struct.file*, i32, i32, i32, i32)* }
	%struct.file_system_type = type { i8*, i32, %struct.super_block* (%struct.super_block*, i8*, i32)*, %struct.module*, %struct.file_system_type*, %struct.list_head }
	%struct.files_struct = type { %struct.IO_APIC_reg_00, %typedef.rwlock_t, i32, i32, i32, %struct.file**, %typedef.__kernel_fd_set*, %typedef.__kernel_fd_set*, %typedef.__kernel_fd_set, %typedef.__kernel_fd_set, [32 x %struct.file*] }
	%struct.fs_disk_quota = type { i8, i8, i16, i32, i64, i64, i64, i64, i64, i64, i32, i32, i16, i16, i32, i64, i64, i64, i32, i16, i16, [8 x i8] }
	%struct.fs_quota_stat = type { i8, i16, i8, %struct.e820entry, %struct.e820entry, i32, i32, i32, i32, i16, i16 }
	%struct.fs_struct = type { %struct.IO_APIC_reg_00, %typedef.rwlock_t, i32, %struct.dentry*, %struct.dentry*, %struct.dentry*, %struct.vfsmount*, %struct.vfsmount*, %struct.vfsmount* }
	%struct.hh_cache = type { %struct.hh_cache*, %struct.IO_APIC_reg_00, i16, i32, i32 (%struct.sk_buff*)*, %typedef.rwlock_t, [32 x i32] }
	%struct.i387_fxsave_struct = type { i16, i16, i16, i16, i32, i32, i32, i32, i32, i32, [32 x i32], [32 x i32], [56 x i32] }
	%struct.iattr = type { i32, i16, i32, i32, i64, i32, i32, i32, i32 }
	%struct.if_dqblk = type { i64, i64, i64, i64, i64, i64, i64, i64, i32 }
	%struct.if_dqinfo = type { i64, i64, i32, i32 }
	%struct.ifmap = type { i32, i32, i16, i8, i8, i8 }
	%struct.ifreq = type { { [16 x i8] }, %typedef.dvd_authinfo }
	%struct.inode = type { %struct.list_head, %struct.list_head, %struct.list_head, %struct.list_head, %struct.list_head, i32, %struct.IO_APIC_reg_00, i16, i16, i16, i32, i32, i16, i64, i32, i32, i32, i32, i32, i32, i32, i16, %struct.semaphore, %struct.rw_semaphore, %struct.semaphore, %struct.inode_operations*, %struct.file_operations*, %struct.super_block*, %struct.__wait_queue_head, %struct.file_lock*, %struct.address_space*, %struct.address_space, [2 x %struct.dquot*], %struct.list_head, %struct.pipe_inode_info*, %struct.block_device*, %struct.char_device*, i32, %struct.dnotify_struct*, i32, i32, i8, %struct.IO_APIC_reg_00, i32, i32, { %struct.ext2_inode_info, %struct.ext3_inode_info, %struct.msdos_inode_info, %struct.iso_inode_info, %struct.nfs_inode_info, %struct._drm_i810_overlay_t, %struct.shmem_inode_info, %struct.proc_inode_info, %struct.socket, %struct.usbdev_inode_info, i8* } }
	%struct.inode_operations = type { i32 (%struct.inode*, %struct.dentry*, i32)*, %struct.dentry* (%struct.inode*, %struct.dentry*)*, i32 (%struct.dentry*, %struct.inode*, %struct.dentry*)*, i32 (%struct.inode*, %struct.dentry*)*, i32 (%struct.inode*, %struct.dentry*, i8*)*, i32 (%struct.inode*, %struct.dentry*, i32)*, i32 (%struct.inode*, %struct.dentry*)*, i32 (%struct.inode*, %struct.dentry*, i32, i32)*, i32 (%struct.inode*, %struct.dentry*, %struct.inode*, %struct.dentry*)*, i32 (%struct.dentry*, i8*, i32)*, i32 (%struct.dentry*, %struct.nameidata*)*, void (%struct.inode*)*, i32 (%struct.inode*, i32)*, i32 (%struct.dentry*)*, i32 (%struct.dentry*, %struct.iattr*)*, i32 (%struct.dentry*, %struct.iattr*)*, i32 (%struct.dentry*, i8*, i8*, i32, i32)*, i32 (%struct.dentry*, i8*, i8*, i32)*, i32 (%struct.dentry*, i8*, i32)*, i32 (%struct.dentry*, i8*)* }
	%struct.iovec = type { i8*, i32 }
	%struct.ip_options = type { i32, i8, i8, i8, i8, i8, i8, i8, i8, [0 x i8] }
	%struct.isapnp_dma = type { i8, i8, %struct.isapnp_resources*, %struct.isapnp_dma* }
	%struct.isapnp_irq = type { i16, i8, i8, %struct.isapnp_resources*, %struct.isapnp_irq* }
	%struct.isapnp_mem = type { i32, i32, i32, i32, i8, i8, %struct.isapnp_resources*, %struct.isapnp_mem* }
	%struct.isapnp_mem32 = type { [17 x i8], %struct.isapnp_resources*, %struct.isapnp_mem32* }
	%struct.isapnp_port = type { i16, i16, i8, i8, i8, i8, %struct.isapnp_resources*, %struct.isapnp_port* }
	%struct.isapnp_resources = type { i16, i16, %struct.isapnp_port*, %struct.isapnp_irq*, %struct.isapnp_dma*, %struct.isapnp_mem*, %struct.isapnp_mem32*, %struct.pci_dev*, %struct.isapnp_resources*, %struct.isapnp_resources* }
	%struct.iso_inode_info = type { i32, i8, [3 x i8], i32, i32 }
	%struct.iw_handler_def = type opaque
	%struct.iw_statistics = type opaque
	%struct.k_sigaction = type { %struct.sigaction }
	%struct.kern_ipc_perm = type { i32, i32, i32, i32, i32, i16, i32 }
	%struct.kiobuf = type { i32, i32, i32, i32, i32, %struct.page**, %struct.buffer_head**, i32*, %struct.IO_APIC_reg_00, i32, void (%struct.kiobuf*)*, %struct.__wait_queue_head }
	%struct.kmem_cache_s = type { %struct.list_head, %struct.list_head, %struct.list_head, i32, i32, i32, %struct.IO_APIC_reg_00, i32, i32, i32, i32, i32, i32, %struct.kmem_cache_s*, i32, i32, void (i8*, %struct.kmem_cache_s*, i32)*, void (i8*, %struct.kmem_cache_s*, i32)*, i32, [20 x i8], %struct.list_head, [32 x %struct._drm_i810_overlay_t*], i32 }
	%struct.linux_binfmt = type { %struct.linux_binfmt*, %struct.module*, i32 (%struct.linux_binprm*, %struct.pt_regs*)*, i32 (%struct.file*)*, i32 (i32, %struct.pt_regs*, %struct.file*)*, i32, i32 (%struct.linux_binprm*, i8*)* }
	%struct.linux_binprm = type { [128 x i8], [32 x %struct.page*], i32, i32, %struct.file*, i32, i32, i32, i32, i32, i32, i32, i8*, i32, i32 }
	%struct.list_head = type { %struct.list_head*, %struct.list_head* }
	%struct.llva_sigcontext = type { %typedef.llva_icontext_t, %typedef.llva_fp_state_t, i32, i32, i32, i32, [1 x i32], i8* }
	%struct.map_segment = type opaque
	%struct.mem_dqblk = type { i32, i32, i64, i32, i32, i32, i32, i32 }
	%struct.mem_dqinfo = type { %struct.quota_format_type*, i32, i32, i32, { %struct.ViceFid } }
	%struct.mm_struct = type { %struct.vm_area_struct*, %struct.rb_root_s, %struct.vm_area_struct*, %struct.IO_APIC_reg_00*, %struct.IO_APIC_reg_00, %struct.IO_APIC_reg_00, i32, %struct.rw_semaphore, %struct.IO_APIC_reg_00, %struct.list_head, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %struct.iovec }
	%struct.module = type { i32, %struct.module*, i8*, i32, %struct.IO_APIC_reg_00, i32, i32, i32, %struct.drm_ctx_priv_map*, %struct.module_ref*, %struct.module_ref*, i32 ()*, void ()*, %struct._drm_i810_overlay_t*, %struct._drm_i810_overlay_t*, %struct.module_persist*, %struct.module_persist*, i32 ()*, i32, i8*, i8*, i8*, i8*, i8* }
	%struct.module_persist = type opaque
	%struct.module_ref = type { %struct.module*, %struct.module*, %struct.module_ref* }
	%struct.msdos_inode_info = type { i32, i32, i32, i32, i32, i32, %struct.inode*, %struct.list_head }
	%struct.msghdr = type { i8*, i32, %struct.iovec*, i32, i8*, i32, i32 }
	%struct.msq_setbuf = type { i32, i32, i32, i16 }
	%struct.nameidata = type { %struct.dentry*, %struct.vfsmount*, %struct.qstr, i32, i32 }
	%struct.namespace = type { %struct.IO_APIC_reg_00, %struct.vfsmount*, %struct.list_head, %struct.rw_semaphore }
	%struct.neigh_ops = type { i32, void (%struct.neighbour*)*, void (%struct.neighbour*, %struct.sk_buff*)*, void (%struct.neighbour*, %struct.sk_buff*)*, i32 (%struct.sk_buff*)*, i32 (%struct.sk_buff*)*, i32 (%struct.sk_buff*)*, i32 (%struct.sk_buff*)* }
	%struct.neigh_parms = type { %struct.neigh_parms*, i32 (%struct.neighbour*)*, %struct.neigh_table*, i32, i8*, i8*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
	%struct.neigh_table = type { %struct.neigh_table*, i32, i32, i32, i32 (i8*, %struct.net_device*)*, i32 (%struct.neighbour*)*, i32 (%struct.pneigh_entry*)*, void (%struct.pneigh_entry*)*, void (%struct.sk_buff*)*, i8*, %struct.neigh_parms, i32, i32, i32, i32, i32, %struct.timer_list, %struct.timer_list, %struct.sk_buff_head, i32, %typedef.rwlock_t, i32, %struct.neigh_parms*, %struct.kmem_cache_s*, %struct.tasklet_struct, %struct.audio_buf_info, [32 x %struct.neighbour*], [16 x %struct.pneigh_entry*] }
	%struct.neighbour = type { %struct.neighbour*, %struct.neigh_table*, %struct.neigh_parms*, %struct.net_device*, i32, i32, i32, i8, i8, i8, i8, %struct.IO_APIC_reg_00, %typedef.rwlock_t, [8 x i8], %struct.hh_cache*, %struct.IO_APIC_reg_00, i32 (%struct.sk_buff*)*, %struct.sk_buff_head, %struct.timer_list, %struct.neigh_ops*, [0 x i8] }
	%struct.net_bridge_port = type opaque
	%struct.net_device = type { [16 x i8], i32, i32, i32, i32, i32, i32, i8, i8, i32, %struct.net_device*, i32 (%struct.net_device*)*, %struct.net_device*, i32, i32, %struct.net_device_stats* (%struct.net_device*)*, %struct.iw_statistics* (%struct.net_device*)*, %struct.iw_handler_def*, i32, i32, i16, i16, i16, i16, i32, i16, i16, i8*, %struct.net_device*, [8 x i8], [8 x i8], i8, %struct.dev_mc_list*, i32, i32, i32, i32, %struct.timer_list, i8*, i8*, i8*, i8*, i8*, %struct.list_head, i32, i32, %struct.Qdisc*, %struct.Qdisc*, %struct.Qdisc*, %struct.Qdisc*, i32, %struct.IO_APIC_reg_00, i32, %struct.IO_APIC_reg_00, %struct.IO_APIC_reg_00, i32, i32, void (%struct.net_device*)*, void (%struct.net_device*)*, i32 (%struct.net_device*)*, i32 (%struct.net_device*)*, i32 (%struct.sk_buff*, %struct.net_device*)*, i32 (%struct.net_device*, i32*)*, i32 (%struct.sk_buff*, %struct.net_device*, i16, i8*, i8*, i32)*, i32 (%struct.sk_buff*)*, void (%struct.net_device*)*, i32 (%struct.net_device*, i8*)*, i32 (%struct.net_device*, %struct.ifreq*, i32)*, i32 (%struct.net_device*, %struct.ifmap*)*, i32 (%struct.neighbour*, %struct.hh_cache*)*, void (%struct.hh_cache*, %struct.net_device*, i8*)*, i32 (%struct.net_device*, i32)*, void (%struct.net_device*)*, void (%struct.net_device*, %struct.vlan_group*)*, void (%struct.net_device*, i16)*, void (%struct.net_device*, i16)*, i32 (%struct.sk_buff*, i8*)*, i32 (%struct.net_device*, %struct.neigh_parms*)*, i32 (%struct.net_device*, %struct.dst_entry*)*, %struct.module*, %struct.net_bridge_port* }
	%struct.net_device_stats = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
	%struct.nf_conntrack = type { %struct.IO_APIC_reg_00, void (%struct.nf_conntrack*)* }
	%struct.nf_ct_info = type { %struct.nf_conntrack* }
	%struct.nfs_fh = type { i16, [64 x i8] }
	%struct.nfs_inode_info = type { i64, %struct.nfs_fh, i16, i32, i64, i64, i64, i32, i32, i32, [2 x i32], %struct.list_head, %struct.list_head, %struct.list_head, %struct.list_head, i32, i32, i32, i32, %struct.rpc_cred* }
	%struct.nfs_lock_info = type { i32, i32, %struct.nlm_host* }
	%struct.nlm_host = type opaque
	%struct.open_request = type { %struct.open_request*, i32, i32, i16, i16, i8, i8, i16, i32, i32, i32, i32, %struct.or_calltable*, %struct.sock*, { %struct.tcp_v4_open_req } }
	%struct.or_calltable = type { i32, i32 (%struct.sock*, %struct.open_request*, %struct.dst_entry*)*, void (%struct.sk_buff*, %struct.open_request*)*, void (%struct.open_request*)*, void (%struct.sk_buff*)* }
	%struct.page = type { %struct.list_head, %struct.address_space*, i32, %struct.page*, %struct.IO_APIC_reg_00, i32, %struct.list_head, %struct.page**, %struct.buffer_head* }
	%struct.pci_bus = type { %struct.list_head, %struct.pci_bus*, %struct.list_head, %struct.list_head, %struct.pci_dev*, [4 x %struct.resource*], %struct.pci_ops*, i8*, %struct.proc_dir_entry*, i8, i8, i8, i8, [48 x i8], i16, i16, i32, i8, i8, i8, i8 }
	%struct.pci_dev = type { %struct.list_head, %struct.list_head, %struct.pci_bus*, %struct.pci_bus*, i8*, %struct.proc_dir_entry*, i32, i16, i16, i16, i16, i32, i8, i8, %struct.pci_driver*, i8*, i64, i32, [4 x i16], [4 x i16], i32, [12 x %struct.resource], [2 x %struct.resource], [2 x %struct.resource], [90 x i8], [8 x i8], i32, i32, i16, i16, i32 (%struct.pci_dev*)*, i32 (%struct.pci_dev*)*, i32 (%struct.pci_dev*)* }
	%struct.pci_device_id = type { i32, i32, i32, i32, i32, i32, i32 }
	%struct.pci_driver = type { %struct.list_head, i8*, %struct.pci_device_id*, i32 (%struct.pci_dev*, %struct.pci_device_id*)*, void (%struct.pci_dev*)*, i32 (%struct.pci_dev*, i32)*, i32 (%struct.pci_dev*, i32)*, i32 (%struct.pci_dev*)*, i32 (%struct.pci_dev*, i32, i32)* }
	%struct.pci_ops = type { i32 (%struct.pci_dev*, i32, i8*)*, i32 (%struct.pci_dev*, i32, i16*)*, i32 (%struct.pci_dev*, i32, i32*)*, i32 (%struct.pci_dev*, i32, i8)*, i32 (%struct.pci_dev*, i32, i16)*, i32 (%struct.pci_dev*, i32, i32)* }
	%struct.pipe_inode_info = type { %struct.__wait_queue_head, i8*, i32, i32, i32, i32, i32, i32, i32, i32 }
	%struct.pneigh_entry = type { %struct.pneigh_entry*, %struct.net_device*, [0 x i8] }
	%struct.poll_table_entry = type { %struct.file*, %struct.__wait_queue, %struct.__wait_queue_head* }
	%struct.poll_table_page = type { %struct.poll_table_page*, %struct.poll_table_entry*, [0 x %struct.poll_table_entry] }
	%struct.poll_table_struct = type { i32, %struct.poll_table_page* }
	%struct.proc_dir_entry = type { i16, i16, i8*, i16, i16, i32, i32, i32, %struct.inode_operations*, %struct.file_operations*, i32 (i8*, i8**, i32, i32)*, %struct.module*, %struct.proc_dir_entry*, %struct.proc_dir_entry*, %struct.proc_dir_entry*, i8*, i32 (i8*, i8**, i32, i32, i32*, i8*)*, i32 (%struct.file*, i8*, i32, i8*)*, %struct.IO_APIC_reg_00, i32, i16 }
	%struct.proc_inode_info = type { %struct.task_struct*, i32, { i32 (%struct.task_struct*, i8*)* }, %struct.file* }
	%struct.proto = type { void (%struct.sock*, i32)*, i32 (%struct.sock*, %struct.sockaddr*, i32)*, i32 (%struct.sock*, i32)*, %struct.sock* (%struct.sock*, i32, i32*)*, i32 (%struct.sock*, i32, i32)*, i32 (%struct.sock*)*, i32 (%struct.sock*)*, void (%struct.sock*, i32)*, i32 (%struct.sock*, i32, i32, i8*, i32)*, i32 (%struct.sock*, i32, i32, i8*, i32*)*, i32 (%struct.sock*, %struct.msghdr*, i32)*, i32 (%struct.sock*, %struct.msghdr*, i32, i32, i32, i32*)*, i32 (%struct.sock*, %struct.sockaddr*, i32)*, i32 (%struct.sock*, %struct.sk_buff*)*, void (%struct.sock*)*, void (%struct.sock*)*, i32 (%struct.sock*, i16)*, [32 x i8], [32 x { i32, [28 x i8] }] }
	%struct.proto_ops = type { i32, i32 (%struct.socket*)*, i32 (%struct.socket*, %struct.sockaddr*, i32)*, i32 (%struct.socket*, %struct.sockaddr*, i32, i32)*, i32 (%struct.socket*, %struct.socket*)*, i32 (%struct.socket*, %struct.socket*, i32)*, i32 (%struct.socket*, %struct.sockaddr*, i32*, i32)*, i32 (%struct.file*, %struct.socket*, %struct.poll_table_struct*)*, i32 (%struct.socket*, i32, i32)*, i32 (%struct.socket*, i32)*, i32 (%struct.socket*, i32)*, i32 (%struct.socket*, i32, i32, i8*, i32)*, i32 (%struct.socket*, i32, i32, i8*, i32*)*, i32 (%struct.socket*, %struct.msghdr*, i32, %struct.scm_cookie*)*, i32 (%struct.socket*, %struct.msghdr*, i32, i32, %struct.scm_cookie*)*, i32 (%struct.file*, %struct.socket*, %struct.vm_area_struct*)*, i32 (%struct.socket*, %struct.page*, i32, i32, i32)* }
	%struct.pt_regs = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
	%struct.qdisc_walker = type { i32, i32, i32, i32 (%struct.Qdisc*, i32, %struct.qdisc_walker*)* }
	%struct.qstr = type { i8*, i32, i32 }
	%struct.quota_format_ops = type { i32 (%struct.super_block*, i32)*, i32 (%struct.super_block*, i32)*, i32 (%struct.super_block*, i32)*, i32 (%struct.super_block*, i32)*, i32 (%struct.dquot*)*, i32 (%struct.dquot*)* }
	%struct.quota_format_type = type { i32, %struct.quota_format_ops*, %struct.module*, %struct.quota_format_type* }
	%struct.quota_info = type { i32, %struct.semaphore, %struct.semaphore, [2 x %struct.file*], [2 x %struct.mem_dqinfo], [2 x %struct.quota_format_ops*] }
	%struct.quotactl_ops = type { i32 (%struct.super_block*, i32, i32, i8*)*, i32 (%struct.super_block*, i32)*, i32 (%struct.super_block*, i32)*, i32 (%struct.super_block*, i32, %struct.if_dqinfo*)*, i32 (%struct.super_block*, i32, %struct.if_dqinfo*)*, i32 (%struct.super_block*, i32, i32, %struct.if_dqblk*)*, i32 (%struct.super_block*, i32, i32, %struct.if_dqblk*)*, i32 (%struct.super_block*, %struct.fs_quota_stat*)*, i32 (%struct.super_block*, i32, i32)*, i32 (%struct.super_block*, i32, i32, %struct.fs_disk_quota*)*, i32 (%struct.super_block*, i32, i32, %struct.fs_disk_quota*)* }
	%struct.rb_node_s = type { %struct.rb_node_s*, i32, %struct.rb_node_s*, %struct.rb_node_s* }
	%struct.rb_root_s = type { %struct.rb_node_s* }
	%struct.resource = type { i8*, i32, i32, i32, %struct.resource*, %struct.resource*, %struct.resource* }
	%struct.revectored_struct = type { [8 x i32] }
	%struct.rpc_auth = type { [8 x %struct.rpc_cred*], i32, i32, i32, i32, i32, %struct.rpc_authops* }
	%struct.rpc_authops = type { i32, i8*, %struct.rpc_auth* (%struct.rpc_clnt*)*, void (%struct.rpc_auth*)*, %struct.rpc_cred* (i32)* }
	%struct.rpc_clnt = type { %struct.IO_APIC_reg_00, %struct.rpc_xprt*, %struct.rpc_procinfo*, i32, i8*, i8*, %struct.rpc_auth*, %struct.rpc_stat*, i32, i32, i32, %struct.rpc_rtt, %struct.msq_setbuf, %struct.rpc_wait_queue, i32, [32 x i8] }
	%struct.rpc_cred = type { %struct.rpc_cred*, %struct.rpc_auth*, %struct.rpc_credops*, i32, %struct.IO_APIC_reg_00, i16, i32, i32 }
	%struct.rpc_credops = type { void (%struct.rpc_cred*)*, i32 (%struct.rpc_cred*, i32)*, i32* (%struct.rpc_task*, i32*, i32)*, i32 (%struct.rpc_task*)*, i32* (%struct.rpc_task*, i32*)* }
	%struct.rpc_message = type { i32, i8*, i8*, %struct.rpc_cred* }
	%struct.rpc_procinfo = type { i8*, i32 (i8*, i32*, i8*)*, i32 (i8*, i32*, i8*)*, i32, i32, i32 }
	%struct.rpc_program = type { i8*, i32, i32, %struct.rpc_version**, %struct.rpc_stat* }
	%struct.rpc_rqst = type { %struct.rpc_xprt*, %struct.rpc_timeout, %struct.xdr_buf, %struct.xdr_buf, %struct.rpc_task*, i32, %struct.rpc_rqst*, i32, i32, %struct.list_head, %struct.xdr_buf, [2 x i32], i32, i32, i32, i32 }
	%struct.rpc_rtt = type { i32, [5 x i32], [5 x i32], %struct.IO_APIC_reg_00 }
	%struct.rpc_stat = type { %struct.rpc_program*, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
	%struct.rpc_task = type { %struct.list_head, i32, %struct.list_head, %struct.rpc_clnt*, %struct.rpc_rqst*, i32, %struct.rpc_wait_queue*, %struct.rpc_message, i32*, i8, i8, i8, void (%struct.rpc_task*)*, void (%struct.rpc_task*)*, void (%struct.rpc_task*)*, void (%struct.rpc_task*)*, void (%struct.rpc_task*)*, i8*, %struct.timer_list, %struct.__wait_queue_head, i32, i16, i8, i32, i16 }
	%struct.rpc_timeout = type { i32, i32, i32, i32, i16, i8 }
	%struct.rpc_version = type { i32, i32, %struct.rpc_procinfo* }
	%struct.rpc_wait_queue = type { %struct.list_head, i8* }
	%struct.rpc_xprt = type { %struct.socket*, %struct.sock*, %struct.rpc_timeout, %struct.sockaddr_in, i32, i32, i32, i32, i32, %struct.rpc_wait_queue, %struct.rpc_wait_queue, %struct.rpc_wait_queue, %struct.rpc_wait_queue, %struct.rpc_rqst*, [16 x %struct.rpc_rqst], i32, i8, i32, i32, i32, i32, i32, i32, %struct.IO_APIC_reg_00, %struct.IO_APIC_reg_00, %struct.rpc_task*, %struct.list_head, void (%struct.sock*, i32)*, void (%struct.sock*)*, void (%struct.sock*)*, %struct.__wait_queue_head }
	%struct.rw_semaphore = type { i32, %struct.IO_APIC_reg_00, %struct.list_head }
	%struct.scm_cookie = type { %struct.ViceFid, %struct.scm_fp_list*, i32 }
	%struct.scm_fp_list = type { i32, [255 x %struct.file*] }
	%struct.sem_array = type { %struct.kern_ipc_perm, i32, i32, %struct._drm_i810_overlay_t*, %struct.sem_queue*, %struct.sem_queue**, %struct.sem_undo*, i32 }
	%struct.sem_queue = type { %struct.sem_queue*, %struct.sem_queue**, %struct.task_struct*, %struct.sem_undo*, i32, i32, %struct.sem_array*, i32, %struct.sembuf*, i32, i32 }
	%struct.sem_undo = type { %struct.sem_undo*, %struct.sem_undo*, i32, i16* }
	%struct.semaphore = type { %struct.IO_APIC_reg_00, i32, %struct.__wait_queue_head }
	%struct.sembuf = type { i16, i16, i16 }
	%struct.seq_file = type { i8*, i32, i32, i32, i64, %struct.semaphore, %struct.seq_operations*, i8* }
	%struct.seq_operations = type { i8* (%struct.seq_file*, i64*)*, void (%struct.seq_file*, i8*)*, i8* (%struct.seq_file*, i8*, i64*)*, i32 (%struct.seq_file*, i8*)* }
	%struct.shmem_inode_info = type { %struct.IO_APIC_reg_00, i32, [16 x %struct.IO_APIC_reg_00], i8**, i32, i32, %struct.list_head, %struct.inode* }
	%struct.sigaction = type { void (i32)*, i32, void ()*, %typedef.__kernel_fsid_t }
	%struct.siginfo = type { i32, i32, i32, { [29 x i32] } }
	%struct.signal_struct = type { %struct.IO_APIC_reg_00, [64 x %struct.k_sigaction], %struct.IO_APIC_reg_00 }
	%struct.sigpending = type { %struct.sigqueue*, %struct.sigqueue**, %typedef.__kernel_fsid_t }
	%struct.sigqueue = type { %struct.sigqueue*, %struct.siginfo }
	%struct.sk_buff = type { %struct.sk_buff*, %struct.sk_buff*, %struct.sk_buff_head*, %struct.sock*, %struct._drm_i810_overlay_t, %struct.net_device*, %struct.net_device*, { i8* }, { i8* }, { i8* }, %struct.dst_entry*, [48 x i8], i32, i32, i32, i8, i8, i8, i8, i32, %struct.IO_APIC_reg_00, i16, i16, i32, i8*, i8*, i8*, i8*, void (%struct.sk_buff*)*, i32, i32, %struct.nf_ct_info*, i32 }
	%struct.sk_buff_head = type { %struct.sk_buff*, %struct.sk_buff*, i32, %struct.IO_APIC_reg_00 }
	%struct.sock = type { i32, i32, i16, i16, i32, %struct.sock*, %struct.sock**, %struct.sock*, %struct.sock**, i8, i8, i16, i16, i8, i8, %struct.IO_APIC_reg_00, %struct.semaphore, i32, %struct.__wait_queue_head*, %struct.dst_entry*, %typedef.rwlock_t, %struct.IO_APIC_reg_00, %struct.sk_buff_head, %struct.IO_APIC_reg_00, %struct.sk_buff_head, %struct.IO_APIC_reg_00, i32, i32, i32, i32, i32, %struct.sock*, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i32, i32, i32, i32, %struct.sock*, { %struct.sk_buff*, %struct.sk_buff* }, %typedef.rwlock_t, %struct.sk_buff_head, %struct.proto*, { %struct.tcp_opt }, i32, i32, i16, i16, i32, i16, i8, i8, %struct.ViceFid, i32, i32, i32, { %struct.unix_opt }, %struct.timer_list, %struct._drm_i810_overlay_t, %struct.socket*, i8*, void (%struct.sock*)*, void (%struct.sock*, i32)*, void (%struct.sock*)*, void (%struct.sock*)*, i32 (%struct.sock*, %struct.sk_buff*)*, void (%struct.sock*)* }
	%struct.sockaddr = type { i16, [14 x i8] }
	%struct.sockaddr_in = type { i16, i16, %struct.IO_APIC_reg_00, [8 x i8] }
	%struct.sockaddr_un = type { i16, [108 x i8] }
	%struct.socket = type { i32, i32, %struct.proto_ops*, %struct.inode*, %struct.fasync_struct*, %struct.file*, %struct.sock*, %struct.__wait_queue_head, i16, i8 }
	%struct.statfs = type { i32, i32, i32, i32, i32, i32, i32, %typedef.__kernel_fsid_t, i32, [6 x i32] }
	%struct.super_block = type { %struct.list_head, i16, i32, i8, i8, i64, %struct.file_system_type*, %struct.super_operations*, %struct.dquot_operations*, %struct.quotactl_ops*, i32, i32, %struct.dentry*, %struct.rw_semaphore, %struct.semaphore, i32, %struct.IO_APIC_reg_00, %struct.list_head, %struct.list_head, %struct.list_head, %struct.block_device*, %struct.list_head, %struct.quota_info, { [115 x i32] }, %struct.semaphore, %struct.semaphore }
	%struct.super_operations = type { %struct.inode* (%struct.super_block*)*, void (%struct.inode*)*, void (%struct.inode*)*, void (%struct.inode*, i8*)*, void (%struct.inode*)*, void (%struct.inode*, i32)*, void (%struct.inode*)*, void (%struct.inode*)*, void (%struct.super_block*)*, void (%struct.super_block*)*, i32 (%struct.super_block*)*, void (%struct.super_block*)*, void (%struct.super_block*)*, i32 (%struct.super_block*, %struct.statfs*)*, i32 (%struct.super_block*, i32*, i8*)*, void (%struct.inode*)*, void (%struct.super_block*)*, %struct.dentry* (%struct.super_block*, i32*, i32, i32, i32)*, i32 (%struct.dentry*, i32*, i32*, i32)*, i32 (%struct.seq_file*, %struct.vfsmount*)* }
	%struct.task_struct = type { i32, i32, i32, %struct.IO_APIC_reg_00, %struct.exec_domain*, i32, i32, i32, i32, i32, i32, %struct.mm_struct*, i32, i32, i32, %struct.list_head, i32, %struct.task_struct*, %struct.task_struct*, %struct.mm_struct*, %struct.list_head, i32, i32, %struct.linux_binfmt*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %struct.task_struct*, %struct.task_struct*, %struct.task_struct*, %struct.task_struct*, %struct.task_struct*, %struct.list_head, %struct.task_struct*, %struct.task_struct**, %struct.__wait_queue_head, %struct.completion*, i32, i32, i32, i32, i32, i32, i32, %struct.timer_list, %struct.audio_buf_info, i32, [32 x i32], [32 x i32], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [32 x i32], i32, i32, i32, i32, %struct.user_struct*, [11 x %struct._drm_i810_overlay_t], i16, [16 x i8], i32, i32, %struct.tty_struct*, i32, %struct.sem_undo*, %struct.sem_queue*, %struct.thread_struct, %struct.fs_struct*, %struct.files_struct*, %struct.namespace*, %struct.IO_APIC_reg_00, %struct.signal_struct*, %typedef.__kernel_fsid_t, %struct.sigpending, i32, i32, i32 (i8*)*, i8*, %typedef.__kernel_fsid_t*, i32, i32, %struct.IO_APIC_reg_00, i8*, %struct.llva_sigcontext*, i32, %struct.task_struct*, i32, %typedef.llva_icontext_t, %typedef.llva_fp_state_t, i32*, i32, i8* }
	%struct.tasklet_struct = type { %struct.tasklet_struct*, i32, %struct.IO_APIC_reg_00, void (i32)*, i32 }
	%struct.tc_stats = type { i64, i32, i32, i32, i32, i32, i32, i32, %struct.IO_APIC_reg_00* }
	%struct.tcf_proto = type { %struct.tcf_proto*, i8*, i32 (%struct.sk_buff*, %struct.tcf_proto*, %struct._drm_i810_overlay_t*)*, i32, i32, i32, %struct.Qdisc*, i8*, %struct.tcf_proto_ops* }
	%struct.tcf_proto_ops = type { %struct.tcf_proto_ops*, [16 x i8], i32 (%struct.sk_buff*, %struct.tcf_proto*, %struct._drm_i810_overlay_t*)*, i32 (%struct.tcf_proto*)*, void (%struct.tcf_proto*)*, i32 (%struct.tcf_proto*, i32)*, void (%struct.tcf_proto*, i32)*, i32 (%struct.tcf_proto*, i32, i32, %struct._agp_version**, i32*)*, i32 (%struct.tcf_proto*, i32)*, void (%struct.tcf_proto*, %struct.tcf_walker*)*, i32 (%struct.tcf_proto*, i32, %struct.sk_buff*, %struct.tcmsg*)* }
	%struct.tcf_walker = type { i32, i32, i32, i32 (%struct.tcf_proto*, i32, %struct.tcf_walker*)* }
	%struct.tcmsg = type { i8, i8, i16, i32, i32, i32, i32 }
	%struct.tcp_func = type { i32 (%struct.sk_buff*)*, void (%struct.sock*, %struct.tcphdr*, i32, %struct.sk_buff*)*, i32 (%struct.sock*)*, i32 (%struct.sock*, %struct.sk_buff*)*, %struct.sock* (%struct.sock*, %struct.sk_buff*, %struct.open_request*, %struct.dst_entry*)*, i32 (%struct.sock*)*, i16, i32 (%struct.sock*, i32, i32, i8*, i32)*, i32 (%struct.sock*, i32, i32, i8*, i32*)*, void (%struct.sock*, %struct.sockaddr*)*, i32 }
	%struct.tcp_listen_opt = type { i8, i32, i32, i32, i32, [512 x %struct.open_request*] }
	%struct.tcp_opt = type { i32, i32, i32, i32, i32, i32, i32, i32, { i8, i8, i8, i8, i32, i32, i32, i16, i16 }, { %struct.sk_buff_head, %struct.task_struct*, %struct.iovec*, i32, i32 }, i32, i32, i32, i32, i16, i16, i16, i8, i8, i8, i8, i8, i8, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i16, i16, i32, i32, i32, %struct.timer_list, %struct.timer_list, %struct.sk_buff_head, %struct.tcp_func*, %struct.sk_buff*, %struct.page*, i32, i32, i32, i32, i32, i32, i8, i8, i8, i8, i8, i8, i8, i8, i32, i32, i32, i32, i16, i8, i8, [1 x %struct._drm_i810_overlay_t], [4 x %struct._drm_i810_overlay_t], i32, i32, i8, i8, i16, i8, i8, i16, i32, i32, i32, i32, i32, i32, i32, i32, i16, i8, i8, i32, %typedef.rwlock_t, %struct.tcp_listen_opt*, %struct.open_request*, %struct.open_request*, i32, i32, i32, i32, i32, i32, i32 }
	%struct.tcp_v4_open_req = type { i32, i32, %struct.ip_options* }
	%struct.tcphdr = type { i16, i16, i32, i32, i16, i16, i16, i16 }
	%struct.termios = type { i32, i32, i32, i32, i8, [19 x i8] }
	%struct.thread_struct = type { i32, i32, i32, i32, i32, [8 x i32], i32, i32, i32, %union.i387_union, %struct.vm86_struct*, i32, i32, i32, i32, i32, [33 x i32] }
	%struct.timer_list = type { %struct.list_head, i32, i32, void (i32)* }
	%struct.tq_struct = type { %struct.list_head, i32, void (i8*)*, i8* }
	%struct.tty_driver = type { i32, i8*, i8*, i32, i16, i16, i16, i16, i16, %struct.termios, i32, i32*, %struct.proc_dir_entry*, %struct.tty_driver*, %struct.tty_struct**, %struct.termios**, %struct.termios**, i8*, i32 (%struct.tty_struct*, %struct.file*)*, void (%struct.tty_struct*, %struct.file*)*, i32 (%struct.tty_struct*, i32, i8*, i32)*, void (%struct.tty_struct*, i8)*, void (%struct.tty_struct*)*, i32 (%struct.tty_struct*)*, i32 (%struct.tty_struct*)*, i32 (%struct.tty_struct*, %struct.file*, i32, i32)*, void (%struct.tty_struct*, %struct.termios*)*, void (%struct.tty_struct*)*, void (%struct.tty_struct*)*, void (%struct.tty_struct*)*, void (%struct.tty_struct*)*, void (%struct.tty_struct*)*, void (%struct.tty_struct*, i32)*, void (%struct.tty_struct*)*, void (%struct.tty_struct*)*, void (%struct.tty_struct*, i32)*, void (%struct.tty_struct*, i8)*, i32 (i8*, i8**, i32, i32, i32*, i8*)*, i32 (%struct.file*, i8*, i32, i8*)*, %struct.tty_driver*, %struct.tty_driver* }
	%struct.tty_flip_buffer = type { %struct.tq_struct, %struct.semaphore, i8*, i8*, i32, i32, [1024 x i8], [1024 x i8], [4 x i8] }
	%struct.tty_ldisc = type { i32, i8*, i32, i32, i32 (%struct.tty_struct*)*, void (%struct.tty_struct*)*, void (%struct.tty_struct*)*, i32 (%struct.tty_struct*)*, i32 (%struct.tty_struct*, %struct.file*, i8*, i32)*, i32 (%struct.tty_struct*, %struct.file*, i8*, i32)*, i32 (%struct.tty_struct*, %struct.file*, i32, i32)*, void (%struct.tty_struct*, %struct.termios*)*, i32 (%struct.tty_struct*, %struct.file*, %struct.poll_table_struct*)*, void (%struct.tty_struct*, i8*, i8*, i32)*, i32 (%struct.tty_struct*)*, void (%struct.tty_struct*)* }
	%struct.tty_struct = type { i32, %struct.tty_driver, %struct.tty_ldisc, %struct.termios*, %struct.termios*, i32, i32, i16, i32, i32, %struct.drm_clip_rect, i8, i8, %struct.tty_struct*, %struct.fasync_struct*, %struct.tty_flip_buffer, i32, i32, %struct.__wait_queue_head, %struct.__wait_queue_head, %struct.tq_struct, i8*, i8*, %struct.list_head, i32, i8, i16, i32, i32, [8 x i32], i8*, i32, i32, i32, [128 x i32], i32, i32, i32, %struct.semaphore, %struct.semaphore, %struct.IO_APIC_reg_00, %struct.tq_struct }
	%struct.unix_address = type { %struct.IO_APIC_reg_00, i32, i32, [0 x %struct.sockaddr_un] }
	%struct.unix_opt = type { %struct.unix_address*, %struct.dentry*, %struct.vfsmount*, %struct.semaphore, %struct.sock*, %struct.sock**, %struct.sock*, %struct.IO_APIC_reg_00, %typedef.rwlock_t, %struct.__wait_queue_head }
	%struct.usb_bus = type opaque
	%struct.usbdev_inode_info = type { %struct.list_head, %struct.list_head, { %struct.usb_bus* } }
	%struct.user_struct = type { %struct.IO_APIC_reg_00, %struct.IO_APIC_reg_00, %struct.IO_APIC_reg_00, %struct.user_struct*, %struct.user_struct**, i32 }
	%struct.vfsmount = type { %struct.list_head, %struct.vfsmount*, %struct.dentry*, %struct.dentry*, %struct.super_block*, %struct.list_head, %struct.list_head, %struct.IO_APIC_reg_00, i32, i8*, %struct.list_head }
	%struct.vlan_group = type opaque
	%struct.vm86_regs = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i16, i16, i32, i32, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16 }
	%struct.vm86_struct = type { %struct.vm86_regs, i32, i32, i32, %struct.revectored_struct, %struct.revectored_struct }
	%struct.vm_area_struct = type { %struct.mm_struct*, i32, i32, %struct.vm_area_struct*, %struct.IO_APIC_reg_00, i32, %struct.rb_node_s, %struct.vm_area_struct*, %struct.vm_area_struct**, %struct.vm_operations_struct*, i32, %struct.file*, i32, i8* }
	%struct.vm_operations_struct = type { void (%struct.vm_area_struct*)*, void (%struct.vm_area_struct*)*, %struct.page* (%struct.vm_area_struct*, i32, i32)* }
	%struct.xdr_buf = type { [1 x %struct.iovec], [1 x %struct.iovec], %struct.page**, i32, i32, i32 }
	%typedef.__kernel_fd_set = type { [32 x i32] }
	%typedef.__kernel_fsid_t = type { [2 x i32] }
	%typedef.dvd_authinfo = type { [2 x i64] }
	%typedef.llva_fp_state_t = type { [7 x i32], [20 x i32] }
	%typedef.llva_icontext_t = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32*, i32 }
	%typedef.rwlock_t = type { %struct.IO_APIC_reg_00, %struct.IO_APIC_reg_00, i32 }
	%typedef.sigset_t = type { [2 x i32] }
	%typedef.socket_lock_t = type { %struct.IO_APIC_reg_00, i32, %struct.__wait_queue_head }
	%union.i387_union = type { %struct.i387_fxsave_struct }

define void @rs_init() {
entry:
	br i1 false, label %loopentry.0.no_exit.0_crit_edge, label %loopentry.0.loopexit.0_crit_edge

loopentry.0:		; No predecessors!
	unreachable

loopentry.0.loopexit.0_crit_edge:		; preds = %entry
	br label %loopexit.0

loopentry.0.no_exit.0_crit_edge:		; preds = %entry
	br label %no_exit.0

no_exit.0:		; preds = %no_exit.0.no_exit.0_crit_edge, %loopentry.0.no_exit.0_crit_edge
	br i1 false, label %no_exit.0.no_exit.0_crit_edge, label %no_exit.0.loopexit.0_crit_edge

no_exit.0.loopexit.0_crit_edge:		; preds = %no_exit.0
	br label %loopexit.0

no_exit.0.no_exit.0_crit_edge:		; preds = %no_exit.0
	br label %no_exit.0

loopexit.0:		; preds = %no_exit.0.loopexit.0_crit_edge, %loopentry.0.loopexit.0_crit_edge
	br i1 false, label %then.0, label %loopexit.0.endif.0_crit_edge

loopexit.0.endif.0_crit_edge:		; preds = %loopexit.0
	br label %endif.0

then.0:		; preds = %loopexit.0
	br i1 false, label %loopentry.1.no_exit.1_crit_edge, label %loopentry.1.loopexit.1_crit_edge

loopentry.1:		; No predecessors!
	unreachable

loopentry.1.loopexit.1_crit_edge:		; preds = %then.0
	br label %loopexit.1

loopentry.1.no_exit.1_crit_edge:		; preds = %then.0
	br label %no_exit.1

no_exit.1:		; preds = %no_exit.1.backedge, %loopentry.1.no_exit.1_crit_edge
	br i1 false, label %shortcirc_next.0, label %no_exit.1.shortcirc_done.0_crit_edge

no_exit.1.shortcirc_done.0_crit_edge:		; preds = %no_exit.1
	br label %shortcirc_done.0

shortcirc_next.0:		; preds = %no_exit.1
	br label %shortcirc_done.0

shortcirc_done.0:		; preds = %shortcirc_next.0, %no_exit.1.shortcirc_done.0_crit_edge
	br i1 false, label %then.1, label %endif.1

then.1:		; preds = %shortcirc_done.0
	br i1 false, label %then.1.no_exit.1_crit_edge, label %then.1.loopexit.1_crit_edge

then.1.loopexit.1_crit_edge:		; preds = %then.1
	br label %loopexit.1

then.1.no_exit.1_crit_edge:		; preds = %then.1
	br label %no_exit.1.backedge

no_exit.1.backedge:		; preds = %endif.1.no_exit.1_crit_edge, %then.1.no_exit.1_crit_edge
	br label %no_exit.1

endif.1:		; preds = %shortcirc_done.0
	br i1 false, label %endif.1.no_exit.1_crit_edge, label %endif.1.loopexit.1_crit_edge

endif.1.loopexit.1_crit_edge:		; preds = %endif.1
	br label %loopexit.1

endif.1.no_exit.1_crit_edge:		; preds = %endif.1
	br label %no_exit.1.backedge

loopexit.1:		; preds = %endif.1.loopexit.1_crit_edge, %then.1.loopexit.1_crit_edge, %loopentry.1.loopexit.1_crit_edge
	br label %endif.0

endif.0:		; preds = %loopexit.1, %loopexit.0.endif.0_crit_edge
	br i1 false, label %then.2, label %endif.0.endif.2_crit_edge

endif.0.endif.2_crit_edge:		; preds = %endif.0
	br label %endif.2

then.2:		; preds = %endif.0
	unreachable

dead_block.0:		; No predecessors!
	br label %endif.2

endif.2:		; preds = %dead_block.0, %endif.0.endif.2_crit_edge
	br i1 false, label %then.3, label %endif.2.endif.3_crit_edge

endif.2.endif.3_crit_edge:		; preds = %endif.2
	br label %endif.3

then.3:		; preds = %endif.2
	unreachable

dead_block.1:		; No predecessors!
	br label %endif.3

endif.3:		; preds = %dead_block.1, %endif.2.endif.3_crit_edge
	br label %loopentry.2

loopentry.2:		; preds = %endif.6, %endif.3
	br i1 false, label %loopentry.2.no_exit.2_crit_edge, label %loopentry.2.loopexit.2_crit_edge

loopentry.2.loopexit.2_crit_edge:		; preds = %loopentry.2
	br label %loopexit.2

loopentry.2.no_exit.2_crit_edge:		; preds = %loopentry.2
	br label %no_exit.2

no_exit.2:		; preds = %then.5.no_exit.2_crit_edge, %loopentry.2.no_exit.2_crit_edge
	br i1 false, label %then.4, label %no_exit.2.endif.4_crit_edge

no_exit.2.endif.4_crit_edge:		; preds = %no_exit.2
	br label %endif.4

then.4:		; preds = %no_exit.2
	br label %endif.4

endif.4:		; preds = %then.4, %no_exit.2.endif.4_crit_edge
	br i1 false, label %shortcirc_next.1, label %endif.4.shortcirc_done.1_crit_edge

endif.4.shortcirc_done.1_crit_edge:		; preds = %endif.4
	br label %shortcirc_done.1

shortcirc_next.1:		; preds = %endif.4
	br i1 false, label %then.i21, label %endif.i

then.i21:		; preds = %shortcirc_next.1
	br label %then.5

then.i21.endif.5_crit_edge:		; No predecessors!
	unreachable

then.i21.then.5_crit_edge:		; No predecessors!
	unreachable

endif.i:		; preds = %shortcirc_next.1
	br label %shortcirc_done.1

__check_region.exit:		; No predecessors!
	unreachable

shortcirc_done.1:		; preds = %endif.i, %endif.4.shortcirc_done.1_crit_edge
	br i1 false, label %shortcirc_done.1.then.5_crit_edge, label %shortcirc_done.1.endif.5_crit_edge

shortcirc_done.1.endif.5_crit_edge:		; preds = %shortcirc_done.1
	br label %endif.5

shortcirc_done.1.then.5_crit_edge:		; preds = %shortcirc_done.1
	br label %then.5

then.5:		; preds = %shortcirc_done.1.then.5_crit_edge, %then.i21
	br i1 false, label %then.5.no_exit.2_crit_edge, label %then.5.loopexit.2_crit_edge

then.5.loopexit.2_crit_edge:		; preds = %then.5
	br label %loopexit.2

then.5.no_exit.2_crit_edge:		; preds = %then.5
	br label %no_exit.2

dead_block_after_continue.0:		; No predecessors!
	unreachable

endif.5:		; preds = %shortcirc_done.1.endif.5_crit_edge
	br i1 false, label %then.6, label %endif.5.endif.6_crit_edge

endif.5.endif.6_crit_edge:		; preds = %endif.5
	br label %endif.6

then.6:		; preds = %endif.5
	br label %endif.6

endif.6:		; preds = %then.6, %endif.5.endif.6_crit_edge
	br label %loopentry.2

loopcont.2:		; No predecessors!
	unreachable

loopexit.2:		; preds = %then.5.loopexit.2_crit_edge, %loopentry.2.loopexit.2_crit_edge
	br label %loopentry.3

loopentry.3:		; preds = %endif.9, %loopexit.2
	br i1 false, label %loopentry.3.no_exit.3_crit_edge, label %loopentry.3.loopexit.3_crit_edge

loopentry.3.loopexit.3_crit_edge:		; preds = %loopentry.3
	br label %loopexit.3

loopentry.3.no_exit.3_crit_edge:		; preds = %loopentry.3
	br label %no_exit.3

no_exit.3:		; preds = %then.7.no_exit.3_crit_edge, %loopentry.3.no_exit.3_crit_edge
	br i1 false, label %then.7, label %no_exit.3.endif.7_crit_edge

no_exit.3.endif.7_crit_edge:		; preds = %no_exit.3
	br label %endif.7

then.7:		; preds = %no_exit.3
	br i1 false, label %then.7.no_exit.3_crit_edge, label %then.7.loopexit.3_crit_edge

then.7.loopexit.3_crit_edge:		; preds = %then.7
	br label %loopexit.3

then.7.no_exit.3_crit_edge:		; preds = %then.7
	br label %no_exit.3

dead_block_after_continue.1:		; No predecessors!
	unreachable

endif.7:		; preds = %no_exit.3.endif.7_crit_edge
	br i1 false, label %shortcirc_next.2, label %endif.7.shortcirc_done.2_crit_edge

endif.7.shortcirc_done.2_crit_edge:		; preds = %endif.7
	br label %shortcirc_done.2

shortcirc_next.2:		; preds = %endif.7
	br label %shortcirc_done.2

shortcirc_done.2:		; preds = %shortcirc_next.2, %endif.7.shortcirc_done.2_crit_edge
	br i1 false, label %shortcirc_next.3, label %shortcirc_done.2.shortcirc_done.3_crit_edge

shortcirc_done.2.shortcirc_done.3_crit_edge:		; preds = %shortcirc_done.2
	br label %shortcirc_done.3

shortcirc_next.3:		; preds = %shortcirc_done.2
	br i1 false, label %shortcirc_next.3.shortcirc_done.4_crit_edge, label %shortcirc_next.4

shortcirc_next.3.shortcirc_done.4_crit_edge:		; preds = %shortcirc_next.3
	br label %shortcirc_done.4

shortcirc_next.4:		; preds = %shortcirc_next.3
	br label %shortcirc_done.4

shortcirc_done.4:		; preds = %shortcirc_next.4, %shortcirc_next.3.shortcirc_done.4_crit_edge
	br label %shortcirc_done.3

shortcirc_done.3:		; preds = %shortcirc_done.4, %shortcirc_done.2.shortcirc_done.3_crit_edge
	br i1 false, label %then.8, label %shortcirc_done.3.endif.8_crit_edge

shortcirc_done.3.endif.8_crit_edge:		; preds = %shortcirc_done.3
	br label %endif.8

then.8:		; preds = %shortcirc_done.3
	br label %endif.8

endif.8:		; preds = %then.8, %shortcirc_done.3.endif.8_crit_edge
	br i1 false, label %then.9, label %else

then.9:		; preds = %endif.8
	br i1 false, label %cond_true.0, label %cond_false.0

cond_true.0:		; preds = %then.9
	br label %cond_continue.0

cond_false.0:		; preds = %then.9
	br label %cond_continue.0

cond_continue.0:		; preds = %cond_false.0, %cond_true.0
	br label %endif.9

else:		; preds = %endif.8
	br i1 false, label %cond_true.1, label %cond_false.1

cond_true.1:		; preds = %else
	br label %cond_continue.1

cond_false.1:		; preds = %else
	br label %cond_continue.1

cond_continue.1:		; preds = %cond_false.1, %cond_true.1
	br label %endif.9

endif.9:		; preds = %cond_continue.1, %cond_continue.0
	br label %loopentry.3

loopcont.3:		; No predecessors!
	unreachable

loopexit.3:		; preds = %then.7.loopexit.3_crit_edge, %loopentry.3.loopexit.3_crit_edge
	br i1 false, label %loopentry.i.i.i2.no_exit.i.i.i4_crit_edge, label %loopentry.i.i.i2.pci_register_driver.exit.i.i_crit_edge

loopentry.i.i.i2:		; No predecessors!
	unreachable

loopentry.i.i.i2.pci_register_driver.exit.i.i_crit_edge:		; preds = %loopexit.3
	br label %pci_register_driver.exit.i.i

loopentry.i.i.i2.no_exit.i.i.i4_crit_edge:		; preds = %loopexit.3
	br label %no_exit.i.i.i4

no_exit.i.i.i4:		; preds = %endif.i.i.i10.no_exit.i.i.i4_crit_edge, %loopentry.i.i.i2.no_exit.i.i.i4_crit_edge
	br i1 false, label %then.i.i.i6, label %no_exit.i.i.i4.endif.i.i.i10_crit_edge

no_exit.i.i.i4.endif.i.i.i10_crit_edge:		; preds = %no_exit.i.i.i4
	br label %endif.i.i.i10

then.i.i.i6:		; preds = %no_exit.i.i.i4
	br i1 false, label %then.0.i.i.i.i, label %else.i.i.i.i

then.0.i.i.i.i:		; preds = %then.i.i.i6
	br i1 false, label %then.1.i.i.i.i, label %endif.1.i.i.i.i

then.1.i.i.i.i:		; preds = %then.0.i.i.i.i
	br label %endif.i.i.i10

endif.1.i.i.i.i:		; preds = %then.0.i.i.i.i
	br i1 false, label %endif.1.i.i.i.i.then.i.i.i.i.i.i_crit_edge, label %endif.1.i.i.i.i.endif.i.i.i.i.i.i_crit_edge

endif.1.i.i.i.i.endif.i.i.i.i.i.i_crit_edge:		; preds = %endif.1.i.i.i.i
	br label %endif.i.i.i.i.i.i

endif.1.i.i.i.i.then.i.i.i.i.i.i_crit_edge:		; preds = %endif.1.i.i.i.i
	br label %then.i.i.i.i.i.i

else.i.i.i.i:		; preds = %then.i.i.i6
	br i1 false, label %endif.0.i.i.i.i.then.i.i.i.i.i.i_crit_edge, label %endif.0.i.i.i.i.endif.i.i.i.i.i.i_crit_edge

endif.0.i.i.i.i:		; No predecessors!
	unreachable

endif.0.i.i.i.i.endif.i.i.i.i.i.i_crit_edge:		; preds = %else.i.i.i.i
	br label %endif.i.i.i.i.i.i

endif.0.i.i.i.i.then.i.i.i.i.i.i_crit_edge:		; preds = %else.i.i.i.i
	br label %then.i.i.i.i.i.i

then.i.i.i.i.i.i:		; preds = %endif.0.i.i.i.i.then.i.i.i.i.i.i_crit_edge, %endif.1.i.i.i.i.then.i.i.i.i.i.i_crit_edge
	br i1 false, label %then.i.i.i.i.i.i.then.2.i.i.i.i_crit_edge, label %then.i.i.i.i.i.i.endif.2.i.i.i.i_crit_edge

then.i.i.i.i.i.i.endif.2.i.i.i.i_crit_edge:		; preds = %then.i.i.i.i.i.i
	br label %endif.2.i.i.i.i

then.i.i.i.i.i.i.then.2.i.i.i.i_crit_edge:		; preds = %then.i.i.i.i.i.i
	br label %then.2.i.i.i.i

endif.i.i.i.i.i.i:		; preds = %endif.0.i.i.i.i.endif.i.i.i.i.i.i_crit_edge, %endif.1.i.i.i.i.endif.i.i.i.i.i.i_crit_edge
	br i1 false, label %dev_probe_lock.exit.i.i.i.i.then.2.i.i.i.i_crit_edge, label %dev_probe_lock.exit.i.i.i.i.endif.2.i.i.i.i_crit_edge

dev_probe_lock.exit.i.i.i.i:		; No predecessors!
	unreachable

dev_probe_lock.exit.i.i.i.i.endif.2.i.i.i.i_crit_edge:		; preds = %endif.i.i.i.i.i.i
	br label %endif.2.i.i.i.i

dev_probe_lock.exit.i.i.i.i.then.2.i.i.i.i_crit_edge:		; preds = %endif.i.i.i.i.i.i
	br label %then.2.i.i.i.i

then.2.i.i.i.i:		; preds = %dev_probe_lock.exit.i.i.i.i.then.2.i.i.i.i_crit_edge, %then.i.i.i.i.i.i.then.2.i.i.i.i_crit_edge
	br label %endif.2.i.i.i.i

endif.2.i.i.i.i:		; preds = %then.2.i.i.i.i, %dev_probe_lock.exit.i.i.i.i.endif.2.i.i.i.i_crit_edge, %then.i.i.i.i.i.i.endif.2.i.i.i.i_crit_edge
	br i1 false, label %then.i.i2.i.i.i.i, label %endif.i.i3.i.i.i.i

then.i.i2.i.i.i.i:		; preds = %endif.2.i.i.i.i
	br label %endif.i.i.i10

endif.i.i3.i.i.i.i:		; preds = %endif.2.i.i.i.i
	br label %endif.i.i.i10

dev_probe_unlock.exit.i.i.i.i:		; No predecessors!
	unreachable

pci_announce_device.exit.i.i.i:		; No predecessors!
	unreachable

endif.i.i.i10:		; preds = %endif.i.i3.i.i.i.i, %then.i.i2.i.i.i.i, %then.1.i.i.i.i, %no_exit.i.i.i4.endif.i.i.i10_crit_edge
	br i1 false, label %endif.i.i.i10.no_exit.i.i.i4_crit_edge, label %endif.i.i.i10.pci_register_driver.exit.i.i_crit_edge

endif.i.i.i10.pci_register_driver.exit.i.i_crit_edge:		; preds = %endif.i.i.i10
	br label %pci_register_driver.exit.i.i

endif.i.i.i10.no_exit.i.i.i4_crit_edge:		; preds = %endif.i.i.i10
	br label %no_exit.i.i.i4

pci_register_driver.exit.i.i:		; preds = %endif.i.i.i10.pci_register_driver.exit.i.i_crit_edge, %loopentry.i.i.i2.pci_register_driver.exit.i.i_crit_edge
	br i1 false, label %then.0.i.i12, label %endif.0.i.i13

then.0.i.i12:		; preds = %pci_register_driver.exit.i.i
	br label %probe_serial_pci.exit

then.0.i.i12.probe_serial_pci.exit_crit_edge:		; No predecessors!
	unreachable

then.0.i.i12.then.i_crit_edge:		; No predecessors!
	br label %then.i

endif.0.i.i13:		; preds = %pci_register_driver.exit.i.i
	br i1 false, label %then.1.i.i14, label %endif.0.i.i13.endif.1.i.i15_crit_edge

endif.0.i.i13.endif.1.i.i15_crit_edge:		; preds = %endif.0.i.i13
	br label %endif.1.i.i15

then.1.i.i14:		; preds = %endif.0.i.i13
	br label %endif.1.i.i15

endif.1.i.i15:		; preds = %then.1.i.i14, %endif.0.i.i13.endif.1.i.i15_crit_edge
	br i1 false, label %loopentry.i8.i.i.no_exit.i9.i.i_crit_edge, label %loopentry.i8.i.i.pci_unregister_driver.exit.i.i_crit_edge

loopentry.i8.i.i:		; No predecessors!
	unreachable

loopentry.i8.i.i.pci_unregister_driver.exit.i.i_crit_edge:		; preds = %endif.1.i.i15
	br label %pci_unregister_driver.exit.i.i

loopentry.i8.i.i.no_exit.i9.i.i_crit_edge:		; preds = %endif.1.i.i15
	br label %no_exit.i9.i.i

no_exit.i9.i.i:		; preds = %endif.0.i.i.i.no_exit.i9.i.i_crit_edge, %loopentry.i8.i.i.no_exit.i9.i.i_crit_edge
	br i1 false, label %then.0.i.i.i, label %no_exit.i9.i.i.endif.0.i.i.i_crit_edge

no_exit.i9.i.i.endif.0.i.i.i_crit_edge:		; preds = %no_exit.i9.i.i
	br label %endif.0.i.i.i

then.0.i.i.i:		; preds = %no_exit.i9.i.i
	br i1 false, label %then.1.i.i.i, label %then.0.i.i.i.endif.1.i.i.i_crit_edge

then.0.i.i.i.endif.1.i.i.i_crit_edge:		; preds = %then.0.i.i.i
	br label %endif.1.i.i.i

then.1.i.i.i:		; preds = %then.0.i.i.i
	br label %endif.1.i.i.i

endif.1.i.i.i:		; preds = %then.1.i.i.i, %then.0.i.i.i.endif.1.i.i.i_crit_edge
	br label %endif.0.i.i.i

endif.0.i.i.i:		; preds = %endif.1.i.i.i, %no_exit.i9.i.i.endif.0.i.i.i_crit_edge
	br i1 false, label %endif.0.i.i.i.no_exit.i9.i.i_crit_edge, label %endif.0.i.i.i.pci_unregister_driver.exit.i.i_crit_edge

endif.0.i.i.i.pci_unregister_driver.exit.i.i_crit_edge:		; preds = %endif.0.i.i.i
	br label %pci_unregister_driver.exit.i.i

endif.0.i.i.i.no_exit.i9.i.i_crit_edge:		; preds = %endif.0.i.i.i
	br label %no_exit.i9.i.i

pci_unregister_driver.exit.i.i:		; preds = %endif.0.i.i.i.pci_unregister_driver.exit.i.i_crit_edge, %loopentry.i8.i.i.pci_unregister_driver.exit.i.i_crit_edge
	br i1 false, label %pci_module_init.exit.i.then.i_crit_edge, label %pci_module_init.exit.i.probe_serial_pci.exit_crit_edge

pci_module_init.exit.i:		; No predecessors!
	unreachable

pci_module_init.exit.i.probe_serial_pci.exit_crit_edge:		; preds = %pci_unregister_driver.exit.i.i
	br label %probe_serial_pci.exit

pci_module_init.exit.i.then.i_crit_edge:		; preds = %pci_unregister_driver.exit.i.i
	br label %then.i

then.i:		; preds = %pci_module_init.exit.i.then.i_crit_edge, %then.0.i.i12.then.i_crit_edge
	br label %probe_serial_pci.exit

probe_serial_pci.exit:		; preds = %then.i, %pci_module_init.exit.i.probe_serial_pci.exit_crit_edge, %then.0.i.i12
	br i1 false, label %then.0.i, label %endif.0.i

then.0.i:		; preds = %probe_serial_pci.exit
	ret void

endif.0.i:		; preds = %probe_serial_pci.exit
	br i1 false, label %loopentry.0.i.no_exit.0.i_crit_edge, label %loopentry.0.i.loopexit.0.i_crit_edge

loopentry.0.i:		; No predecessors!
	unreachable

loopentry.0.i.loopexit.0.i_crit_edge:		; preds = %endif.0.i
	br label %loopexit.0.i

loopentry.0.i.no_exit.0.i_crit_edge:		; preds = %endif.0.i
	br label %no_exit.0.i

no_exit.0.i:		; preds = %loopcont.0.i.no_exit.0.i_crit_edge, %loopentry.0.i.no_exit.0.i_crit_edge
	br i1 false, label %then.1.i, label %endif.1.i

then.1.i:		; preds = %no_exit.0.i
	br label %loopcont.0.i

endif.1.i:		; preds = %no_exit.0.i
	br i1 false, label %loopentry.1.i.no_exit.1.i_crit_edge, label %loopentry.1.i.loopexit.1.i_crit_edge

loopentry.1.i:		; No predecessors!
	unreachable

loopentry.1.i.loopexit.1.i_crit_edge:		; preds = %endif.1.i
	br label %loopexit.1.i

loopentry.1.i.no_exit.1.i_crit_edge:		; preds = %endif.1.i
	br label %no_exit.1.i

no_exit.1.i:		; preds = %endif.2.i.no_exit.1.i_crit_edge, %loopentry.1.i.no_exit.1.i_crit_edge
	br i1 false, label %shortcirc_next.0.i, label %no_exit.1.i.shortcirc_done.0.i_crit_edge

no_exit.1.i.shortcirc_done.0.i_crit_edge:		; preds = %no_exit.1.i
	br label %shortcirc_done.0.i

shortcirc_next.0.i:		; preds = %no_exit.1.i
	br label %shortcirc_done.0.i

shortcirc_done.0.i:		; preds = %shortcirc_next.0.i, %no_exit.1.i.shortcirc_done.0.i_crit_edge
	br i1 false, label %then.2.i, label %endif.2.i

then.2.i:		; preds = %shortcirc_done.0.i
	br i1 false, label %then.2.i.then.3.i_crit_edge, label %then.2.i.else.i_crit_edge

then.2.i.else.i_crit_edge:		; preds = %then.2.i
	br label %else.i

then.2.i.then.3.i_crit_edge:		; preds = %then.2.i
	br label %then.3.i

endif.2.i:		; preds = %shortcirc_done.0.i
	br i1 false, label %endif.2.i.no_exit.1.i_crit_edge, label %endif.2.i.loopexit.1.i_crit_edge

endif.2.i.loopexit.1.i_crit_edge:		; preds = %endif.2.i
	br label %loopexit.1.i

endif.2.i.no_exit.1.i_crit_edge:		; preds = %endif.2.i
	br label %no_exit.1.i

loopexit.1.i:		; preds = %endif.2.i.loopexit.1.i_crit_edge, %loopentry.1.i.loopexit.1.i_crit_edge
	br i1 false, label %loopexit.1.i.then.3.i_crit_edge, label %loopexit.1.i.else.i_crit_edge

loopexit.1.i.else.i_crit_edge:		; preds = %loopexit.1.i
	br label %else.i

loopexit.1.i.then.3.i_crit_edge:		; preds = %loopexit.1.i
	br label %then.3.i

then.3.i:		; preds = %loopexit.1.i.then.3.i_crit_edge, %then.2.i.then.3.i_crit_edge
	br i1 false, label %shortcirc_next.1.i, label %then.3.i.shortcirc_done.1.i_crit_edge

then.3.i.shortcirc_done.1.i_crit_edge:		; preds = %then.3.i
	br label %shortcirc_done.1.i

shortcirc_next.1.i:		; preds = %then.3.i
	br label %shortcirc_done.1.i

shortcirc_done.1.i:		; preds = %shortcirc_next.1.i, %then.3.i.shortcirc_done.1.i_crit_edge
	br i1 false, label %then.4.i, label %endif.4.i

then.4.i:		; preds = %shortcirc_done.1.i
	br label %endif.3.i

endif.4.i:		; preds = %shortcirc_done.1.i
	br label %endif.3.i

else.i:		; preds = %loopexit.1.i.else.i_crit_edge, %then.2.i.else.i_crit_edge
	br i1 false, label %shortcirc_next.0.i.i, label %else.i.shortcirc_done.0.i.i_crit_edge

else.i.shortcirc_done.0.i.i_crit_edge:		; preds = %else.i
	br label %shortcirc_done.0.i.i

shortcirc_next.0.i.i:		; preds = %else.i
	br label %shortcirc_done.0.i.i

shortcirc_done.0.i.i:		; preds = %shortcirc_next.0.i.i, %else.i.shortcirc_done.0.i.i_crit_edge
	br i1 false, label %shortcirc_next.1.i.i, label %shortcirc_done.0.i.i.shortcirc_done.1.i.i_crit_edge

shortcirc_done.0.i.i.shortcirc_done.1.i.i_crit_edge:		; preds = %shortcirc_done.0.i.i
	br label %shortcirc_done.1.i.i

shortcirc_next.1.i.i:		; preds = %shortcirc_done.0.i.i
	br i1 false, label %loopentry.i.i2.i.no_exit.i.i3.i_crit_edge, label %loopentry.i.i2.i.loopexit.i.i.i_crit_edge

loopentry.i.i2.i:		; No predecessors!
	unreachable

loopentry.i.i2.i.loopexit.i.i.i_crit_edge:		; preds = %shortcirc_next.1.i.i
	br label %loopexit.i.i.i

loopentry.i.i2.i.no_exit.i.i3.i_crit_edge:		; preds = %shortcirc_next.1.i.i
	br label %no_exit.i.i3.i

no_exit.i.i3.i:		; preds = %endif.i.i.i.no_exit.i.i3.i_crit_edge, %loopentry.i.i2.i.no_exit.i.i3.i_crit_edge
	br i1 false, label %shortcirc_next.0.i.i.i, label %no_exit.i.i3.i.shortcirc_done.0.i.i.i_crit_edge

no_exit.i.i3.i.shortcirc_done.0.i.i.i_crit_edge:		; preds = %no_exit.i.i3.i
	br label %shortcirc_done.0.i.i.i

shortcirc_next.0.i.i.i:		; preds = %no_exit.i.i3.i
	br label %shortcirc_done.0.i.i.i

shortcirc_done.0.i.i.i:		; preds = %shortcirc_next.0.i.i.i, %no_exit.i.i3.i.shortcirc_done.0.i.i.i_crit_edge
	br i1 false, label %shortcirc_next.1.i.i.i, label %shortcirc_done.0.i.i.i.shortcirc_done.1.i.i.i_crit_edge

shortcirc_done.0.i.i.i.shortcirc_done.1.i.i.i_crit_edge:		; preds = %shortcirc_done.0.i.i.i
	br label %shortcirc_done.1.i.i.i

shortcirc_next.1.i.i.i:		; preds = %shortcirc_done.0.i.i.i
	br label %shortcirc_done.1.i.i.i

shortcirc_done.1.i.i.i:		; preds = %shortcirc_next.1.i.i.i, %shortcirc_done.0.i.i.i.shortcirc_done.1.i.i.i_crit_edge
	br i1 false, label %then.i.i.i, label %endif.i.i.i

then.i.i.i:		; preds = %shortcirc_done.1.i.i.i
	br label %then.0.i.i

then.i.i.i.endif.0.i.i_crit_edge:		; No predecessors!
	unreachable

then.i.i.i.then.0.i.i_crit_edge:		; No predecessors!
	unreachable

endif.i.i.i:		; preds = %shortcirc_done.1.i.i.i
	br i1 false, label %endif.i.i.i.no_exit.i.i3.i_crit_edge, label %endif.i.i.i.loopexit.i.i.i_crit_edge

endif.i.i.i.loopexit.i.i.i_crit_edge:		; preds = %endif.i.i.i
	br label %loopexit.i.i.i

endif.i.i.i.no_exit.i.i3.i_crit_edge:		; preds = %endif.i.i.i
	br label %no_exit.i.i3.i

loopexit.i.i.i:		; preds = %endif.i.i.i.loopexit.i.i.i_crit_edge, %loopentry.i.i2.i.loopexit.i.i.i_crit_edge
	br label %shortcirc_done.1.i.i

check_compatible_id.exit.i.i:		; No predecessors!
	unreachable

shortcirc_done.1.i.i:		; preds = %loopexit.i.i.i, %shortcirc_done.0.i.i.shortcirc_done.1.i.i_crit_edge
	br i1 false, label %shortcirc_done.1.i.i.then.0.i.i_crit_edge, label %shortcirc_done.1.i.i.endif.0.i.i_crit_edge

shortcirc_done.1.i.i.endif.0.i.i_crit_edge:		; preds = %shortcirc_done.1.i.i
	br label %endif.0.i.i

shortcirc_done.1.i.i.then.0.i.i_crit_edge:		; preds = %shortcirc_done.1.i.i
	br label %then.0.i.i

then.0.i.i:		; preds = %shortcirc_done.1.i.i.then.0.i.i_crit_edge, %then.i.i.i
	br label %then.5.i

then.0.i.i.endif.5.i_crit_edge:		; No predecessors!
	unreachable

then.0.i.i.then.5.i_crit_edge:		; No predecessors!
	unreachable

endif.0.i.i:		; preds = %shortcirc_done.1.i.i.endif.0.i.i_crit_edge
	br i1 false, label %endif.0.i.i.shortcirc_done.2.i.i_crit_edge, label %shortcirc_next.2.i.i

endif.0.i.i.shortcirc_done.2.i.i_crit_edge:		; preds = %endif.0.i.i
	br label %shortcirc_done.2.i.i

shortcirc_next.2.i.i:		; preds = %endif.0.i.i
	br label %shortcirc_done.2.i.i

shortcirc_done.2.i.i:		; preds = %shortcirc_next.2.i.i, %endif.0.i.i.shortcirc_done.2.i.i_crit_edge
	br i1 false, label %then.1.i.i, label %endif.1.i.i

then.1.i.i:		; preds = %shortcirc_done.2.i.i
	br label %then.5.i

then.1.i.i.endif.5.i_crit_edge:		; No predecessors!
	unreachable

then.1.i.i.then.5.i_crit_edge:		; No predecessors!
	unreachable

endif.1.i.i:		; preds = %shortcirc_done.2.i.i
	br i1 false, label %loopentry.0.i7.i.no_exit.0.i8.i_crit_edge, label %loopentry.0.i7.i.loopexit.0.i11.i_crit_edge

loopentry.0.i7.i:		; No predecessors!
	unreachable

loopentry.0.i7.i.loopexit.0.i11.i_crit_edge:		; preds = %endif.1.i.i
	br label %loopexit.0.i11.i

loopentry.0.i7.i.no_exit.0.i8.i_crit_edge:		; preds = %endif.1.i.i
	br label %no_exit.0.i8.i

no_exit.0.i8.i:		; preds = %loopexit.1.i.i.no_exit.0.i8.i_crit_edge, %loopentry.0.i7.i.no_exit.0.i8.i_crit_edge
	br i1 false, label %loopentry.1.i9.i.no_exit.1.i10.i_crit_edge, label %loopentry.1.i9.i.loopexit.1.i.i_crit_edge

loopentry.1.i9.i:		; No predecessors!
	unreachable

loopentry.1.i9.i.loopexit.1.i.i_crit_edge:		; preds = %no_exit.0.i8.i
	br label %loopexit.1.i.i

loopentry.1.i9.i.no_exit.1.i10.i_crit_edge:		; preds = %no_exit.0.i8.i
	br label %no_exit.1.i10.i

no_exit.1.i10.i:		; preds = %endif.2.i.i.no_exit.1.i10.i_crit_edge, %loopentry.1.i9.i.no_exit.1.i10.i_crit_edge
	br i1 false, label %shortcirc_next.3.i.i, label %no_exit.1.i10.i.shortcirc_done.3.i.i_crit_edge

no_exit.1.i10.i.shortcirc_done.3.i.i_crit_edge:		; preds = %no_exit.1.i10.i
	br label %shortcirc_done.3.i.i

shortcirc_next.3.i.i:		; preds = %no_exit.1.i10.i
	br i1 false, label %shortcirc_next.3.i.i.shortcirc_done.4.i.i_crit_edge, label %shortcirc_next.4.i.i

shortcirc_next.3.i.i.shortcirc_done.4.i.i_crit_edge:		; preds = %shortcirc_next.3.i.i
	br label %shortcirc_done.4.i.i

shortcirc_next.4.i.i:		; preds = %shortcirc_next.3.i.i
	br label %shortcirc_done.4.i.i

shortcirc_done.4.i.i:		; preds = %shortcirc_next.4.i.i, %shortcirc_next.3.i.i.shortcirc_done.4.i.i_crit_edge
	br i1 false, label %shortcirc_done.4.i.i.shortcirc_done.5.i.i_crit_edge, label %shortcirc_next.5.i.i

shortcirc_done.4.i.i.shortcirc_done.5.i.i_crit_edge:		; preds = %shortcirc_done.4.i.i
	br label %shortcirc_done.5.i.i

shortcirc_next.5.i.i:		; preds = %shortcirc_done.4.i.i
	%tmp.68.i.i = icmp eq i16 0, 1000		; <i1> [#uses=1]
	br label %shortcirc_done.5.i.i

shortcirc_done.5.i.i:		; preds = %shortcirc_next.5.i.i, %shortcirc_done.4.i.i.shortcirc_done.5.i.i_crit_edge
	%shortcirc_val.4.i.i = phi i1 [ true, %shortcirc_done.4.i.i.shortcirc_done.5.i.i_crit_edge ], [ %tmp.68.i.i, %shortcirc_next.5.i.i ]		; <i1> [#uses=1]
	br label %shortcirc_done.3.i.i

shortcirc_done.3.i.i:		; preds = %shortcirc_done.5.i.i, %no_exit.1.i10.i.shortcirc_done.3.i.i_crit_edge
	%shortcirc_val.5.i.i = phi i1 [ false, %no_exit.1.i10.i.shortcirc_done.3.i.i_crit_edge ], [ %shortcirc_val.4.i.i, %shortcirc_done.5.i.i ]		; <i1> [#uses=1]
	br i1 %shortcirc_val.5.i.i, label %then.2.i.i, label %endif.2.i.i

then.2.i.i:		; preds = %shortcirc_done.3.i.i
	%port.2.i.i.8.lcssa20 = phi %struct.isapnp_port* [ null, %shortcirc_done.3.i.i ]		; <%struct.isapnp_port*> [#uses=0]
	br label %endif.5.i

then.2.i.i.endif.5.i_crit_edge:		; No predecessors!
	unreachable

then.2.i.i.then.5.i_crit_edge:		; No predecessors!
	unreachable

endif.2.i.i:		; preds = %shortcirc_done.3.i.i
	br i1 false, label %endif.2.i.i.no_exit.1.i10.i_crit_edge, label %endif.2.i.i.loopexit.1.i.i_crit_edge

endif.2.i.i.loopexit.1.i.i_crit_edge:		; preds = %endif.2.i.i
	br label %loopexit.1.i.i

endif.2.i.i.no_exit.1.i10.i_crit_edge:		; preds = %endif.2.i.i
	br label %no_exit.1.i10.i

loopexit.1.i.i:		; preds = %endif.2.i.i.loopexit.1.i.i_crit_edge, %loopentry.1.i9.i.loopexit.1.i.i_crit_edge
	br i1 false, label %loopexit.1.i.i.no_exit.0.i8.i_crit_edge, label %loopexit.1.i.i.loopexit.0.i11.i_crit_edge

loopexit.1.i.i.loopexit.0.i11.i_crit_edge:		; preds = %loopexit.1.i.i
	br label %loopexit.0.i11.i

loopexit.1.i.i.no_exit.0.i8.i_crit_edge:		; preds = %loopexit.1.i.i
	br label %no_exit.0.i8.i

loopexit.0.i11.i:		; preds = %loopexit.1.i.i.loopexit.0.i11.i_crit_edge, %loopentry.0.i7.i.loopexit.0.i11.i_crit_edge
	br i1 false, label %serial_pnp_guess_board.exit.i.then.5.i_crit_edge, label %serial_pnp_guess_board.exit.i.endif.5.i_crit_edge

serial_pnp_guess_board.exit.i:		; No predecessors!
	unreachable

serial_pnp_guess_board.exit.i.endif.5.i_crit_edge:		; preds = %loopexit.0.i11.i
	br label %endif.5.i

serial_pnp_guess_board.exit.i.then.5.i_crit_edge:		; preds = %loopexit.0.i11.i
	br label %then.5.i

then.5.i:		; preds = %serial_pnp_guess_board.exit.i.then.5.i_crit_edge, %then.1.i.i, %then.0.i.i
	br label %loopcont.0.i

endif.5.i:		; preds = %serial_pnp_guess_board.exit.i.endif.5.i_crit_edge, %then.2.i.i
	br label %endif.3.i

endif.3.i:		; preds = %endif.5.i, %endif.4.i, %then.4.i
	br i1 false, label %then.6.i, label %endif.3.i.endif.6.i_crit_edge

endif.3.i.endif.6.i_crit_edge:		; preds = %endif.3.i
	br label %endif.6.i

then.6.i:		; preds = %endif.3.i
	br label %loopentry.0.i.i

loopentry.0.i.i:		; preds = %endif.i.i, %then.6.i
	br i1 false, label %loopentry.0.i.i.no_exit.0.i.i_crit_edge, label %loopentry.0.i.i.loopexit.0.i.i_crit_edge

loopentry.0.i.i.loopexit.0.i.i_crit_edge:		; preds = %loopentry.0.i.i
	br label %loopexit.0.i.i

loopentry.0.i.i.no_exit.0.i.i_crit_edge:		; preds = %loopentry.0.i.i
	br label %no_exit.0.i.i

no_exit.0.i.i:		; preds = %clear_bit195.exit.i.i.no_exit.0.i.i_crit_edge, %loopentry.0.i.i.no_exit.0.i.i_crit_edge
	br i1 false, label %then.i.i, label %endif.i.i

then.i.i:		; preds = %no_exit.0.i.i
	br label %loopentry.i.i.i

loopentry.i.i.i:		; preds = %no_exit.i.i.i, %then.i.i
	br i1 false, label %no_exit.i.i.i, label %clear_bit195.exit.i.i

no_exit.i.i.i:		; preds = %loopentry.i.i.i
	br label %loopentry.i.i.i

clear_bit195.exit.i.i:		; preds = %loopentry.i.i.i
	br i1 false, label %clear_bit195.exit.i.i.no_exit.0.i.i_crit_edge, label %clear_bit195.exit.i.i.loopexit.0.i.i_crit_edge

clear_bit195.exit.i.i.loopexit.0.i.i_crit_edge:		; preds = %clear_bit195.exit.i.i
	br label %loopexit.0.i.i

clear_bit195.exit.i.i.no_exit.0.i.i_crit_edge:		; preds = %clear_bit195.exit.i.i
	br label %no_exit.0.i.i

endif.i.i:		; preds = %no_exit.0.i.i
	br label %loopentry.0.i.i

loopexit.0.i.i:		; preds = %clear_bit195.exit.i.i.loopexit.0.i.i_crit_edge, %loopentry.0.i.i.loopexit.0.i.i_crit_edge
	br i1 false, label %loopentry.1.i.i.no_exit.1.i.i_crit_edge, label %loopentry.1.i.i.avoid_irq_share.exit.i_crit_edge

loopentry.1.i.i:		; No predecessors!
	unreachable

loopentry.1.i.i.avoid_irq_share.exit.i_crit_edge:		; preds = %loopexit.0.i.i
	br label %avoid_irq_share.exit.i

loopentry.1.i.i.no_exit.1.i.i_crit_edge:		; preds = %loopexit.0.i.i
	br label %no_exit.1.i.i

no_exit.1.i.i:		; preds = %loopexit.2.i.i.no_exit.1.i.i_crit_edge, %loopentry.1.i.i.no_exit.1.i.i_crit_edge
	br i1 false, label %loopentry.2.i.i.no_exit.2.i.i_crit_edge, label %loopentry.2.i.i.loopexit.2.i.i_crit_edge

loopentry.2.i.i:		; No predecessors!
	unreachable

loopentry.2.i.i.loopexit.2.i.i_crit_edge:		; preds = %no_exit.1.i.i
	br label %loopexit.2.i.i

loopentry.2.i.i.no_exit.2.i.i_crit_edge:		; preds = %no_exit.1.i.i
	br label %no_exit.2.i.i

no_exit.2.i.i:		; preds = %no_exit.2.i.i.no_exit.2.i.i_crit_edge, %loopentry.2.i.i.no_exit.2.i.i_crit_edge
	br i1 false, label %no_exit.2.i.i.no_exit.2.i.i_crit_edge, label %no_exit.2.i.i.loopexit.2.i.i_crit_edge

no_exit.2.i.i.loopexit.2.i.i_crit_edge:		; preds = %no_exit.2.i.i
	br label %loopexit.2.i.i

no_exit.2.i.i.no_exit.2.i.i_crit_edge:		; preds = %no_exit.2.i.i
	br label %no_exit.2.i.i

loopexit.2.i.i:		; preds = %no_exit.2.i.i.loopexit.2.i.i_crit_edge, %loopentry.2.i.i.loopexit.2.i.i_crit_edge
	br i1 false, label %loopexit.2.i.i.no_exit.1.i.i_crit_edge, label %loopexit.2.i.i.avoid_irq_share.exit.i_crit_edge

loopexit.2.i.i.avoid_irq_share.exit.i_crit_edge:		; preds = %loopexit.2.i.i
	br label %avoid_irq_share.exit.i

loopexit.2.i.i.no_exit.1.i.i_crit_edge:		; preds = %loopexit.2.i.i
	br label %no_exit.1.i.i

avoid_irq_share.exit.i:		; preds = %loopexit.2.i.i.avoid_irq_share.exit.i_crit_edge, %loopentry.1.i.i.avoid_irq_share.exit.i_crit_edge
	br label %endif.6.i

endif.6.i:		; preds = %avoid_irq_share.exit.i, %endif.3.i.endif.6.i_crit_edge
	br label %loopcont.0.i

loopcont.0.i:		; preds = %endif.6.i, %then.5.i, %then.1.i
	br i1 false, label %loopcont.0.i.no_exit.0.i_crit_edge, label %loopcont.0.i.loopexit.0.i_crit_edge

loopcont.0.i.loopexit.0.i_crit_edge:		; preds = %loopcont.0.i
	br label %loopexit.0.i

loopcont.0.i.no_exit.0.i_crit_edge:		; preds = %loopcont.0.i
	br label %no_exit.0.i

loopexit.0.i:		; preds = %loopcont.0.i.loopexit.0.i_crit_edge, %loopentry.0.i.loopexit.0.i_crit_edge
	ret void

probe_serial_pnp.exit:		; No predecessors!
	unreachable

after_ret:		; No predecessors!
	ret void

return:		; No predecessors!
	unreachable
}
