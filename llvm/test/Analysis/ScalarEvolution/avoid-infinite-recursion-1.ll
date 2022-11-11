; RUN: opt < %s -passes='require<iv-users>'
; PR4538

; ModuleID = 'bugpoint-reduced-simplified.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-freebsd8.0"
module asm ".ident\09\22$FreeBSD: head/sys/kern/vfs_subr.c 195285 2009-07-02 14:19:33Z jamie $\22"
module asm ".section set_pcpu, \22aw\22, @progbits"
module asm ".previous"
	%0 = type <{ [40 x i8] }>		; type %0
	%1 = type <{ %struct.vm_object*, %struct.vm_object** }>		; type %1
	%2 = type <{ %struct.vm_object* }>		; type %2
	%3 = type <{ %struct.vm_page*, %struct.vm_page** }>		; type %3
	%4 = type <{ %struct.pv_entry*, %struct.pv_entry** }>		; type %4
	%5 = type <{ %struct.vm_reserv* }>		; type %5
	%6 = type <{ %struct.bufobj*, %struct.bufobj** }>		; type %6
	%7 = type <{ %struct.proc*, %struct.proc** }>		; type %7
	%8 = type <{ %struct.thread*, %struct.thread** }>		; type %8
	%9 = type <{ %struct.prison*, %struct.prison** }>		; type %9
	%10 = type <{ %struct.prison* }>		; type %10
	%11 = type <{ %struct.task* }>		; type %11
	%12 = type <{ %struct.osd*, %struct.osd** }>		; type %12
	%13 = type <{ %struct.proc* }>		; type %13
	%14 = type <{ %struct.ksiginfo*, %struct.ksiginfo** }>		; type %14
	%15 = type <{ %struct.pv_chunk*, %struct.pv_chunk** }>		; type %15
	%16 = type <{ %struct.pgrp*, %struct.pgrp** }>		; type %16
	%17 = type <{ %struct.knote*, %struct.knote** }>		; type %17
	%18 = type <{ %struct.ktr_request*, %struct.ktr_request** }>		; type %18
	%19 = type <{ %struct.mqueue_notifier* }>		; type %19
	%20 = type <{ %struct.turnstile* }>		; type %20
	%21 = type <{ %struct.namecache* }>		; type %21
	%22 = type <{ %struct.namecache*, %struct.namecache** }>		; type %22
	%23 = type <{ %struct.lockf*, %struct.lockf** }>		; type %23
	%24 = type <{ %struct.lockf_entry*, %struct.lockf_entry** }>		; type %24
	%25 = type <{ %struct.lockf_edge*, %struct.lockf_edge** }>		; type %25
	%struct.__siginfo = type <{ i32, i32, i32, i32, i32, i32, i8*, %union.sigval, %0 }>
	%struct.__sigset = type <{ [4 x i32] }>
	%struct.acl = type <{ i32, i32, [4 x i32], [254 x %struct.acl_entry] }>
	%struct.acl_entry = type <{ i32, i32, i32, i16, i16 }>
	%struct.au_mask = type <{ i32, i32 }>
	%struct.au_tid_addr = type <{ i32, i32, [4 x i32] }>
	%struct.auditinfo_addr = type <{ i32, %struct.au_mask, %struct.au_tid_addr, i32, i64 }>
	%struct.bintime = type <{ i64, i64 }>
	%struct.buf = type <{ %struct.bufobj*, i64, i8*, i8*, i32, i8, i8, i8, i8, i64, i64, void (%struct.buf*)*, i64, i64, %struct.buflists, %struct.buf*, %struct.buf*, i32, i8, i8, i8, i8, %struct.buflists, i16, i8, i8, i32, i8, i8, i8, i8, i8, i8, i8, i8, %struct.lock, i64, i64, i8*, i32, i8, i8, i8, i8, i64, %struct.vnode*, i32, i32, %struct.ucred*, %struct.ucred*, i8*, %union.pager_info, i8, i8, i8, i8, %union.anon, [32 x %struct.vm_page*], i32, i8, i8, i8, i8, %struct.workhead, i8*, i8*, i8*, i32, i8, i8, i8, i8 }>
	%struct.buf_ops = type <{ i8*, i32 (%struct.buf*)*, void (%struct.bufobj*, %struct.buf*)*, i32 (%struct.bufobj*, i32)*, void (%struct.bufobj*, %struct.buf*)* }>
	%struct.buflists = type <{ %struct.buf*, %struct.buf** }>
	%struct.bufobj = type <{ %struct.mtx, %struct.bufv, %struct.bufv, i64, i32, i8, i8, i8, i8, %struct.buf_ops*, i32, i8, i8, i8, i8, %struct.vm_object*, %6, i8*, %struct.vnode* }>
	%struct.bufv = type <{ %struct.buflists, %struct.buf*, i32, i8, i8, i8, i8 }>
	%struct.callout = type <{ %union.anon, i32, i8, i8, i8, i8, i8*, void (i8*)*, %struct.lock_object*, i32, i32 }>
	%struct.cdev_privdata = type opaque
	%struct.cluster_save = type <{ i64, i64, i8*, i32, i8, i8, i8, i8, %struct.buf** }>
	%struct.componentname = type <{ i64, i64, %struct.thread*, %struct.ucred*, i32, i8, i8, i8, i8, i8*, i8*, i64, i64 }>
	%struct.cpuset = type opaque
	%struct.cv = type <{ i8*, i32, i8, i8, i8, i8 }>
	%struct.fid = type <{ i16, i16, [16 x i8] }>
	%struct.file = type <{ i8*, %struct.fileops*, %struct.ucred*, %struct.vnode*, i16, i16, i32, i32, i32, i64, %struct.cdev_privdata*, i64, i8* }>
	%struct.filedesc = type opaque
	%struct.filedesc_to_leader = type opaque
	%struct.fileops = type <{ i32 (%struct.file*, %struct.uio*, %struct.ucred*, i32, %struct.thread*)*, i32 (%struct.file*, %struct.uio*, %struct.ucred*, i32, %struct.thread*)*, i32 (%struct.file*, i64, %struct.ucred*, %struct.thread*)*, i32 (%struct.file*, i64, i8*, %struct.ucred*, %struct.thread*)*, i32 (%struct.file*, i32, %struct.ucred*, %struct.thread*)*, i32 (%struct.file*, %struct.knote*)*, i32 (%struct.file*, %struct.stat*, %struct.ucred*, %struct.thread*)*, i32 (%struct.file*, %struct.thread*)*, i32, i8, i8, i8, i8 }>
	%struct.filterops = type <{ i32, i8, i8, i8, i8, i32 (%struct.knote*)*, void (%struct.knote*)*, i32 (%struct.knote*, i64)* }>
	%struct.flock = type <{ i64, i64, i32, i16, i16, i32, i8, i8, i8, i8 }>
	%struct.freelst = type <{ %struct.vnode*, %struct.vnode** }>
	%struct.fsid = type <{ [2 x i32] }>
	%struct.in6_addr = type opaque
	%struct.in_addr = type opaque
	%struct.inode = type opaque
	%struct.iovec = type <{ i8*, i64 }>
	%struct.itimers = type opaque
	%struct.itimerval = type <{ %struct.bintime, %struct.bintime }>
	%struct.kaioinfo = type opaque
	%struct.kaudit_record = type opaque
	%struct.kdtrace_proc = type opaque
	%struct.kdtrace_thread = type opaque
	%struct.kevent = type <{ i64, i16, i16, i32, i64, i8* }>
	%struct.klist = type <{ %struct.knote* }>
	%struct.knlist = type <{ %struct.klist, void (i8*)*, void (i8*)*, void (i8*)*, void (i8*)*, i8* }>
	%struct.knote = type <{ %struct.klist, %struct.klist, %struct.knlist*, %17, %struct.kqueue*, %struct.kevent, i32, i32, i64, %union.sigval, %struct.filterops*, i8* }>
	%struct.kqueue = type opaque
	%struct.ksiginfo = type <{ %14, %struct.__siginfo, i32, i8, i8, i8, i8, %struct.sigqueue* }>
	%struct.ktr_request = type opaque
	%struct.label = type opaque
	%struct.lock = type <{ %struct.lock_object, i64, i32, i32 }>
	%struct.lock_list_entry = type opaque
	%struct.lock_object = type <{ i8*, i32, i32, %struct.witness* }>
	%struct.lock_owner = type opaque
	%struct.lock_profile_object = type opaque
	%struct.lockf = type <{ %23, %struct.mtx, %struct.lockf_entry_list, %struct.lockf_entry_list, i32, i8, i8, i8, i8 }>
	%struct.lockf_edge = type <{ %25, %25, %struct.lockf_entry*, %struct.lockf_entry* }>
	%struct.lockf_edge_list = type <{ %struct.lockf_edge* }>
	%struct.lockf_entry = type <{ i16, i16, i8, i8, i8, i8, i64, i64, %struct.lock_owner*, %struct.vnode*, %struct.inode*, %struct.task*, %24, %struct.lockf_edge_list, %struct.lockf_edge_list, i32, i8, i8, i8, i8 }>
	%struct.lockf_entry_list = type <{ %struct.lockf_entry* }>
	%struct.lpohead = type <{ %struct.lock_profile_object* }>
	%struct.md_page = type <{ %4 }>
	%struct.mdproc = type <{ %struct.cv*, %struct.system_segment_descriptor }>
	%struct.mdthread = type <{ i32, i8, i8, i8, i8, i64 }>
	%struct.mntarg = type opaque
	%struct.mntlist = type <{ %struct.mount*, %struct.mount** }>
	%struct.mount = type <{ %struct.mtx, i32, i8, i8, i8, i8, %struct.mntlist, %struct.vfsops*, %struct.vfsconf*, %struct.vnode*, %struct.vnode*, i32, i8, i8, i8, i8, %struct.freelst, i32, i32, i32, i32, i32, i32, %struct.vfsoptlist*, %struct.vfsoptlist*, i32, i8, i8, i8, i8, %struct.statfs, %struct.ucred*, i8*, i64, i32, i8, i8, i8, i8, %struct.netexport*, %struct.label*, i32, i32, i32, i32, %struct.thread*, i8*, %struct.lock }>
	%struct.mqueue_notifier = type opaque
	%struct.mtx = type <{ %struct.lock_object, i64 }>
	%struct.namecache = type opaque
	%struct.netexport = type opaque
	%struct.nlminfo = type opaque
	%struct.osd = type <{ i32, i8, i8, i8, i8, i8**, %12 }>
	%struct.p_sched = type opaque
	%struct.pargs = type <{ i32, i32, [1 x i8], i8, i8, i8 }>
	%struct.pcb = type opaque
	%struct.pgrp = type <{ %16, %13, %struct.session*, %struct.sigiolst, i32, i32, %struct.mtx }>
	%struct.plimit = type opaque
	%struct.pmap = type <{ %struct.mtx, i64*, %15, i32, i8, i8, i8, i8, %struct.bintime, %struct.vm_page* }>
	%struct.prison = type <{ %9, i32, i32, i32, i32, %10, %9, %struct.prison*, %struct.mtx, %struct.task, %struct.osd, %struct.cpuset*, %struct.vnet*, %struct.vnode*, i32, i32, %struct.in_addr*, %struct.in6_addr*, [4 x i8*], i32, i32, i32, i32, i32, [5 x i32], i64, [256 x i8], [1024 x i8], [256 x i8], [256 x i8], [64 x i8] }>
	%struct.proc = type <{ %7, %8, %struct.mtx, %struct.ucred*, %struct.filedesc*, %struct.filedesc_to_leader*, %struct.pstats*, %struct.plimit*, %struct.callout, %struct.sigacts*, i32, i32, i32, i8, i8, i8, i8, %7, %7, %struct.proc*, %7, %13, %struct.mtx, %struct.ksiginfo*, %struct.sigqueue, i32, i8, i8, i8, i8, %struct.vmspace*, i32, i8, i8, i8, i8, %struct.itimerval, %struct.rusage, %struct.rusage_ext, %struct.rusage_ext, i32, i32, i32, i8, i8, i8, i8, %struct.vnode*, %struct.ucred*, %struct.vnode*, i32, i8, i8, i8, i8, %struct.sigiolst, i32, i32, i64, i32, i32, i8, i8, i8, i8, i8, i8, i8, i8, %struct.nlminfo*, %struct.kaioinfo*, %struct.thread*, i32, i8, i8, i8, i8, %struct.thread*, i32, i32, %struct.itimers*, i32, i32, [20 x i8], i8, i8, i8, i8, %struct.pgrp*, %struct.sysentvec*, %struct.pargs*, i64, i8, i8, i8, i8, i32, i16, i8, i8, i8, i8, i8, i8, %struct.knlist, i32, i8, i8, i8, i8, %struct.mdproc, %struct.callout, i16, i8, i8, i8, i8, i8, i8, %struct.proc*, %struct.proc*, i8*, %struct.label*, %struct.p_sched*, %18, %19, %struct.kdtrace_proc*, %struct.cv }>
	%struct.pstats = type opaque
	%struct.pv_chunk = type <{ %struct.pmap*, %15, [3 x i64], [2 x i64], [168 x %struct.pv_entry] }>
	%struct.pv_entry = type <{ i64, %4 }>
	%struct.rusage = type <{ %struct.bintime, %struct.bintime, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64 }>
	%struct.rusage_ext = type <{ i64, i64, i64, i64, i64, i64, i64 }>
	%struct.selfd = type opaque
	%struct.selfdlist = type <{ %struct.selfd*, %struct.selfd** }>
	%struct.selinfo = type <{ %struct.selfdlist, %struct.knlist, %struct.mtx* }>
	%struct.seltd = type opaque
	%struct.session = type <{ i32, i8, i8, i8, i8, %struct.proc*, %struct.vnode*, %struct.tty*, i32, [24 x i8], i8, i8, i8, i8, %struct.mtx }>
	%struct.shmmap_state = type opaque
	%struct.sigacts = type <{ [128 x void (i32)*], [128 x %struct.__sigset], %struct.__sigset, %struct.__sigset, %struct.__sigset, %struct.__sigset, %struct.__sigset, %struct.__sigset, %struct.__sigset, %struct.__sigset, %struct.__sigset, %struct.__sigset, i32, i32, %struct.mtx }>
	%struct.sigaltstack = type <{ i8*, i64, i32, i8, i8, i8, i8 }>
	%struct.sigio = type <{ %union.sigval, %struct.sigiolst, %struct.sigio**, %struct.ucred*, i32, i8, i8, i8, i8 }>
	%struct.sigiolst = type <{ %struct.sigio* }>
	%struct.sigqueue = type <{ %struct.__sigset, %struct.__sigset, %14, %struct.proc*, i32, i8, i8, i8, i8 }>
	%struct.sleepqueue = type opaque
	%struct.sockaddr = type opaque
	%struct.stat = type <{ i32, i32, i16, i16, i32, i32, i32, %struct.bintime, %struct.bintime, %struct.bintime, i64, i64, i32, i32, i32, i32, %struct.bintime }>
	%struct.statfs = type <{ i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, [10 x i64], i32, i32, %struct.fsid, [80 x i8], [16 x i8], [88 x i8], [88 x i8] }>
	%struct.sysctl_req = type <{ %struct.thread*, i32, i8, i8, i8, i8, i8*, i64, i64, i32 (%struct.sysctl_req*, i8*, i64)*, i8*, i64, i64, i32 (%struct.sysctl_req*, i8*, i64)*, i64, i32, i8, i8, i8, i8 }>
	%struct.sysentvec = type opaque
	%struct.system_segment_descriptor = type <{ i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8 }>
	%struct.task = type <{ %11, i16, i16, i8, i8, i8, i8, void (i8*, i32)*, i8* }>
	%struct.td_sched = type opaque
	%struct.thread = type <{ %struct.mtx*, %struct.proc*, %8, %8, %8, %8, %struct.cpuset*, %struct.seltd*, %struct.sleepqueue*, %struct.turnstile*, %struct.umtx_q*, i32, i8, i8, i8, i8, %struct.sigqueue, i32, i32, i32, i32, i32, i8, i8, i8, i8, i8*, i8*, i8, i8, i8, i8, i16, i16, i16, i8, i8, i8, i8, i8, i8, %struct.turnstile*, i8*, %20, %struct.lock_list_entry*, i32, i32, %struct.ucred*, i32, i32, %struct.rusage, i64, i64, i32, i32, i32, i32, i32, %struct.__sigset, %struct.__sigset, i32, %struct.sigaltstack, i32, i8, i8, i8, i8, i64, i32, [20 x i8], %struct.file*, i32, i32, %struct.osd, i8, i8, i8, i8, i8, i8, i8, i8, %struct.pcb*, i32, i8, i8, i8, i8, [2 x i64], %struct.callout, %struct.trapframe*, %struct.vm_object*, i64, i32, i8, i8, i8, i8, %struct.vm_object*, i64, i32, i32, %struct.mdthread, %struct.td_sched*, %struct.kaudit_record*, i32, i8, i8, i8, i8, [2 x %struct.lpohead], %struct.kdtrace_thread*, i32, i8, i8, i8, i8, %struct.vnet*, i8* }>
	%struct.trapframe = type <{ i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32, i16, i16, i64, i32, i16, i16, i64, i64, i64, i64, i64, i64 }>
	%struct.tty = type opaque
	%struct.turnstile = type opaque
	%struct.ucred = type <{ i32, i32, i32, i32, i32, i32, i32, i8, i8, i8, i8, %struct.uidinfo*, %struct.uidinfo*, %struct.prison*, %struct.vimage*, i32, i8, i8, i8, i8, [2 x i8*], %struct.label*, %struct.auditinfo_addr, i32*, i32, i8, i8, i8, i8 }>
	%struct.uidinfo = type opaque
	%struct.uio = type <{ %struct.iovec*, i32, i8, i8, i8, i8, i64, i64, i32, i32, %struct.thread* }>
	%struct.umtx_q = type opaque
	%struct.vattr = type <{ i32, i16, i16, i32, i32, i32, i8, i8, i8, i8, i64, i64, i64, %struct.bintime, %struct.bintime, %struct.bintime, %struct.bintime, i64, i64, i32, i8, i8, i8, i8, i64, i64, i32, i8, i8, i8, i8, i64 }>
	%struct.vfsconf = type <{ i32, [16 x i8], i8, i8, i8, i8, %struct.vfsops*, i32, i32, i32, i8, i8, i8, i8, %struct.vfsoptdecl*, %struct.vfsconfhead }>
	%struct.vfsconfhead = type <{ %struct.vfsconf*, %struct.vfsconf** }>
	%struct.vfsops = type <{ i32 (%struct.mount*)*, i32 (%struct.mntarg*, i8*, i32)*, i32 (%struct.mount*, i32)*, i32 (%struct.mount*, i32, %struct.vnode**)*, i32 (%struct.mount*, i32, i32, i8*)*, i32 (%struct.mount*, %struct.statfs*)*, i32 (%struct.mount*, i32)*, i32 (%struct.mount*, i32, i32, %struct.vnode**)*, i32 (%struct.mount*, %struct.fid*, %struct.vnode**)*, i32 (%struct.mount*, %struct.sockaddr*, i32*, %struct.ucred**, i32*, i32**)*, i32 (%struct.vfsconf*)*, i32 (%struct.vfsconf*)*, i32 (%struct.mount*, i32, %struct.vnode*, i32, i8*)*, i32 (%struct.mount*, i32, %struct.sysctl_req*)*, void (%struct.mount*)* }>
	%struct.vfsopt = type <{ %struct.vfsoptlist, i8*, i8*, i32, i32, i32, i8, i8, i8, i8 }>
	%struct.vfsoptdecl = type opaque
	%struct.vfsoptlist = type <{ %struct.vfsopt*, %struct.vfsopt** }>
	%struct.vimage = type opaque
	%struct.vm_map = type <{ %struct.vm_map_entry, %struct.mtx, %struct.mtx, i32, i8, i8, i8, i8, i64, i32, i8, i8, i8, i8, %struct.vm_map_entry*, %struct.pmap*, %struct.vm_map_entry* }>
	%struct.vm_map_entry = type <{ %struct.vm_map_entry*, %struct.vm_map_entry*, %struct.vm_map_entry*, %struct.vm_map_entry*, i64, i64, i64, i64, i64, %union.sigval, i64, i32, i8, i8, i8, i8, i32, i8, i8, i8, i8, i64, %struct.uidinfo* }>
	%struct.vm_object = type <{ %struct.mtx, %1, %2, %1, %3, %struct.vm_page*, i64, i32, i32, i32, i8, i8, i16, i16, i16, i32, %struct.vm_object*, i64, %1, %5, %struct.vm_page*, i8*, %union.anon, %struct.uidinfo*, i64 }>
	%struct.vm_page = type <{ %3, %3, %struct.vm_page*, %struct.vm_page*, %struct.vm_object*, i64, i64, %struct.md_page, i8, i8, i16, i8, i8, i16, i32, i16, i16, i8, i8, i8, i8, i8, i8, i8, i8 }>
	%struct.vm_reserv = type opaque
	%struct.vmspace = type <{ %struct.vm_map, %struct.shmmap_state*, i64, i64, i64, i64, i8*, i8*, i8*, i32, i8, i8, i8, i8, %struct.pmap }>
	%struct.vnet = type opaque
	%struct.vnode = type <{ i32, i8, i8, i8, i8, i8*, %struct.vop_vector*, i8*, %struct.mount*, %struct.freelst, %union.sigval, %struct.freelst, i32, i8, i8, i8, i8, %21, %22, %struct.namecache*, i64, i64, i64, i32, i8, i8, i8, i8, %struct.lock, %struct.mtx, %struct.lock*, i32, i32, i64, i64, i32, i8, i8, i8, i8, %struct.freelst, %struct.bufobj, %struct.vpollinfo*, %struct.label*, %struct.lockf* }>
	%struct.vnodeop_desc = type <{ i8*, i32, i8, i8, i8, i8, i32 (%struct.vop_generic_args*)*, i32*, i32, i32, i32, i32 }>
	%struct.vop_access_args = type <{ %struct.vop_generic_args, %struct.vnode*, i32, i8, i8, i8, i8, %struct.ucred*, %struct.thread* }>
	%struct.vop_aclcheck_args = type <{ %struct.vop_generic_args, %struct.vnode*, i32, i8, i8, i8, i8, %struct.acl*, %struct.ucred*, %struct.thread* }>
	%struct.vop_advlock_args = type <{ %struct.vop_generic_args, %struct.vnode*, i8*, i32, i8, i8, i8, i8, %struct.flock*, i32, i8, i8, i8, i8 }>
	%struct.vop_advlockasync_args = type <{ %struct.vop_generic_args, %struct.vnode*, i8*, i32, i8, i8, i8, i8, %struct.flock*, i32, i8, i8, i8, i8, %struct.task*, i8** }>
	%struct.vop_bmap_args = type <{ %struct.vop_generic_args, %struct.vnode*, i64, %struct.bufobj**, i64*, i32*, i32* }>
	%struct.vop_cachedlookup_args = type <{ %struct.vop_generic_args, %struct.vnode*, %struct.vnode**, %struct.componentname* }>
	%struct.vop_create_args = type <{ %struct.vop_generic_args, %struct.vnode*, %struct.vnode**, %struct.componentname*, %struct.vattr* }>
	%struct.vop_deleteextattr_args = type <{ %struct.vop_generic_args, %struct.vnode*, i32, i8, i8, i8, i8, i8*, %struct.ucred*, %struct.thread* }>
	%struct.vop_fsync_args = type <{ %struct.vop_generic_args, %struct.vnode*, i32, i8, i8, i8, i8, %struct.thread* }>
	%struct.vop_generic_args = type <{ %struct.vnodeop_desc* }>
	%struct.vop_getattr_args = type <{ %struct.vop_generic_args, %struct.vnode*, %struct.vattr*, %struct.ucred* }>
	%struct.vop_getextattr_args = type <{ %struct.vop_generic_args, %struct.vnode*, i32, i8, i8, i8, i8, i8*, %struct.uio*, i64*, %struct.ucred*, %struct.thread* }>
	%struct.vop_getpages_args = type <{ %struct.vop_generic_args, %struct.vnode*, %struct.vm_page**, i32, i32, i64 }>
	%struct.vop_getwritemount_args = type <{ %struct.vop_generic_args, %struct.vnode*, %struct.mount** }>
	%struct.vop_inactive_args = type <{ %struct.vop_generic_args, %struct.vnode*, %struct.thread* }>
	%struct.vop_ioctl_args = type <{ %struct.vop_generic_args, %struct.vnode*, i64, i8*, i32, i8, i8, i8, i8, %struct.ucred*, %struct.thread* }>
	%struct.vop_islocked_args = type <{ %struct.vop_generic_args, %struct.vnode* }>
	%struct.vop_kqfilter_args = type <{ %struct.vop_generic_args, %struct.vnode*, %struct.knote* }>
	%struct.vop_link_args = type <{ %struct.vop_generic_args, %struct.vnode*, %struct.vnode*, %struct.componentname* }>
	%struct.vop_listextattr_args = type <{ %struct.vop_generic_args, %struct.vnode*, i32, i8, i8, i8, i8, %struct.uio*, i64*, %struct.ucred*, %struct.thread* }>
	%struct.vop_lock1_args = type <{ %struct.vop_generic_args, %struct.vnode*, i32, i8, i8, i8, i8, i8*, i32, i8, i8, i8, i8 }>
	%struct.vop_open_args = type <{ %struct.vop_generic_args, %struct.vnode*, i32, i8, i8, i8, i8, %struct.ucred*, %struct.thread*, %struct.file* }>
	%struct.vop_openextattr_args = type <{ %struct.vop_generic_args, %struct.vnode*, %struct.ucred*, %struct.thread* }>
	%struct.vop_pathconf_args = type <{ %struct.vop_generic_args, %struct.vnode*, i32, i8, i8, i8, i8, i64* }>
	%struct.vop_putpages_args = type <{ %struct.vop_generic_args, %struct.vnode*, %struct.vm_page**, i32, i32, i32*, i64 }>
	%struct.vop_read_args = type <{ %struct.vop_generic_args, %struct.vnode*, %struct.uio*, i32, i8, i8, i8, i8, %struct.ucred* }>
	%struct.vop_readdir_args = type <{ %struct.vop_generic_args, %struct.vnode*, %struct.uio*, %struct.ucred*, i32*, i32*, i64** }>
	%struct.vop_readlink_args = type <{ %struct.vop_generic_args, %struct.vnode*, %struct.uio*, %struct.ucred* }>
	%struct.vop_reallocblks_args = type <{ %struct.vop_generic_args, %struct.vnode*, %struct.cluster_save* }>
	%struct.vop_rename_args = type <{ %struct.vop_generic_args, %struct.vnode*, %struct.vnode*, %struct.componentname*, %struct.vnode*, %struct.vnode*, %struct.componentname* }>
	%struct.vop_revoke_args = type <{ %struct.vop_generic_args, %struct.vnode*, i32, i8, i8, i8, i8 }>
	%struct.vop_setextattr_args = type <{ %struct.vop_generic_args, %struct.vnode*, i32, i8, i8, i8, i8, i8*, %struct.uio*, %struct.ucred*, %struct.thread* }>
	%struct.vop_setlabel_args = type <{ %struct.vop_generic_args, %struct.vnode*, %struct.label*, %struct.ucred*, %struct.thread* }>
	%struct.vop_strategy_args = type <{ %struct.vop_generic_args, %struct.vnode*, %struct.buf* }>
	%struct.vop_symlink_args = type <{ %struct.vop_generic_args, %struct.vnode*, %struct.vnode**, %struct.componentname*, %struct.vattr*, i8* }>
	%struct.vop_vector = type <{ %struct.vop_vector*, i32 (%struct.vop_generic_args*)*, i32 (%struct.vop_islocked_args*)*, i32 (%struct.vop_cachedlookup_args*)*, i32 (%struct.vop_cachedlookup_args*)*, i32 (%struct.vop_create_args*)*, i32 (%struct.vop_whiteout_args*)*, i32 (%struct.vop_create_args*)*, i32 (%struct.vop_open_args*)*, i32 (%struct.vop_access_args*)*, i32 (%struct.vop_access_args*)*, i32 (%struct.vop_access_args*)*, i32 (%struct.vop_getattr_args*)*, i32 (%struct.vop_getattr_args*)*, i32 (%struct.vop_islocked_args*)*, i32 (%struct.vop_read_args*)*, i32 (%struct.vop_read_args*)*, i32 (%struct.vop_ioctl_args*)*, i32 (%struct.vop_access_args*)*, i32 (%struct.vop_kqfilter_args*)*, i32 (%struct.vop_revoke_args*)*, i32 (%struct.vop_fsync_args*)*, i32 (%struct.vop_link_args*)*, i32 (%struct.vop_link_args*)*, i32 (%struct.vop_rename_args*)*, i32 (%struct.vop_create_args*)*, i32 (%struct.vop_link_args*)*, i32 (%struct.vop_symlink_args*)*, i32 (%struct.vop_readdir_args*)*, i32 (%struct.vop_readlink_args*)*, i32 (%struct.vop_inactive_args*)*, i32 (%struct.vop_inactive_args*)*, i32 (%struct.vop_lock1_args*)*, i32 (%struct.vop_revoke_args*)*, i32 (%struct.vop_bmap_args*)*, i32 (%struct.vop_strategy_args*)*, i32 (%struct.vop_getwritemount_args*)*, i32 (%struct.vop_islocked_args*)*, i32 (%struct.vop_pathconf_args*)*, i32 (%struct.vop_advlock_args*)*, i32 (%struct.vop_advlockasync_args*)*, i32 (%struct.vop_reallocblks_args*)*, i32 (%struct.vop_getpages_args*)*, i32 (%struct.vop_putpages_args*)*, i32 (%struct.vop_aclcheck_args*)*, i32 (%struct.vop_aclcheck_args*)*, i32 (%struct.vop_aclcheck_args*)*, i32 (%struct.vop_access_args*)*, i32 (%struct.vop_getextattr_args*)*, i32 (%struct.vop_listextattr_args*)*, i32 (%struct.vop_openextattr_args*)*, i32 (%struct.vop_deleteextattr_args*)*, i32 (%struct.vop_setextattr_args*)*, i32 (%struct.vop_setlabel_args*)*, i32 (%struct.vop_vptofh_args*)*, i32 (%struct.vop_vptocnp_args*)* }>
	%struct.vop_vptocnp_args = type <{ %struct.vop_generic_args, %struct.vnode*, %struct.vnode**, %struct.ucred*, i8*, i32* }>
	%struct.vop_vptofh_args = type <{ %struct.vop_generic_args, %struct.vnode*, %struct.fid* }>
	%struct.vop_whiteout_args = type <{ %struct.vop_generic_args, %struct.vnode*, %struct.componentname*, i32, i8, i8, i8, i8 }>
	%struct.vpollinfo = type <{ %struct.mtx, %struct.selinfo, i16, i16, i8, i8, i8, i8 }>
	%struct.witness = type opaque
	%struct.workhead = type <{ %struct.worklist* }>
	%struct.worklist = type opaque
	%union.anon = type <{ [16 x i8] }>
	%union.pager_info = type <{ [4 x i8] }>
	%union.sigval = type <{ [8 x i8] }>

define i32 @vlrureclaim(%struct.mount* %mp) nounwind {
entry:
	br i1 undef, label %if.then11, label %do.end

if.then11:		; preds = %entry
	br label %do.end

do.end:		; preds = %if.then11, %entry
	br label %while.cond.outer

while.cond.outer:		; preds = %while.cond.outer.backedge, %do.end
	%count.0.ph = phi i32 [ undef, %do.end ], [ undef, %while.cond.outer.backedge ]		; <i32> [#uses=1]
	br label %while.cond

while.cond:		; preds = %next_iter, %while.cond.outer
	%count.0 = phi i32 [ %dec, %next_iter ], [ %count.0.ph, %while.cond.outer ]		; <i32> [#uses=2]
	%cmp21 = icmp eq i32 %count.0, 0		; <i1> [#uses=1]
	br i1 %cmp21, label %do.body288.loopexit4, label %while.body

while.body:		; preds = %while.cond
	br label %while.cond27

while.cond27:		; preds = %while.body36, %while.body
	br i1 undef, label %do.body288.loopexit, label %land.rhs

land.rhs:		; preds = %while.cond27
	br i1 undef, label %while.body36, label %while.end

while.body36:		; preds = %land.rhs
	br label %while.cond27

while.end:		; preds = %land.rhs
	br i1 undef, label %do.body288.loopexit4, label %do.body46

do.body46:		; preds = %while.end
	br i1 undef, label %if.else64, label %if.then53

if.then53:		; preds = %do.body46
	br label %if.end72

if.else64:		; preds = %do.body46
	br label %if.end72

if.end72:		; preds = %if.else64, %if.then53
	%dec = add i32 %count.0, -1		; <i32> [#uses=2]
	br i1 undef, label %next_iter, label %if.end111

if.end111:		; preds = %if.end72
	br i1 undef, label %lor.lhs.false, label %do.body145

lor.lhs.false:		; preds = %if.end111
	br i1 undef, label %lor.lhs.false122, label %do.body145

lor.lhs.false122:		; preds = %lor.lhs.false
	br i1 undef, label %lor.lhs.false128, label %do.body145

lor.lhs.false128:		; preds = %lor.lhs.false122
	br i1 undef, label %do.body162, label %land.lhs.true

land.lhs.true:		; preds = %lor.lhs.false128
	br i1 undef, label %do.body145, label %do.body162

do.body145:		; preds = %land.lhs.true, %lor.lhs.false122, %lor.lhs.false, %if.end111
	br i1 undef, label %if.then156, label %next_iter

if.then156:		; preds = %do.body145
	br label %next_iter

do.body162:		; preds = %land.lhs.true, %lor.lhs.false128
	br i1 undef, label %if.then173, label %do.end177

if.then173:		; preds = %do.body162
	br label %do.end177

do.end177:		; preds = %if.then173, %do.body162
	br i1 undef, label %do.body185, label %if.then182

if.then182:		; preds = %do.end177
	br label %next_iter_mntunlocked

do.body185:		; preds = %do.end177
	br i1 undef, label %if.then196, label %do.end202

if.then196:		; preds = %do.body185
	br label %do.end202

do.end202:		; preds = %if.then196, %do.body185
	br i1 undef, label %lor.lhs.false207, label %if.then231

lor.lhs.false207:		; preds = %do.end202
	br i1 undef, label %lor.lhs.false214, label %if.then231

lor.lhs.false214:		; preds = %lor.lhs.false207
	br i1 undef, label %do.end236, label %land.lhs.true221

land.lhs.true221:		; preds = %lor.lhs.false214
	br i1 undef, label %if.then231, label %do.end236

if.then231:		; preds = %land.lhs.true221, %lor.lhs.false207, %do.end202
	br label %next_iter_mntunlocked

do.end236:		; preds = %land.lhs.true221, %lor.lhs.false214
	br label %next_iter_mntunlocked

next_iter_mntunlocked:		; preds = %do.end236, %if.then231, %if.then182
	br i1 undef, label %yield, label %do.body269

next_iter:		; preds = %if.then156, %do.body145, %if.end72
	%rem2482 = and i32 %dec, 255		; <i32> [#uses=1]
	%cmp249 = icmp eq i32 %rem2482, 0		; <i1> [#uses=1]
	br i1 %cmp249, label %do.body253, label %while.cond

do.body253:		; preds = %next_iter
	br i1 undef, label %if.then264, label %yield

if.then264:		; preds = %do.body253
	br label %yield

yield:		; preds = %if.then264, %do.body253, %next_iter_mntunlocked
	br label %do.body269

do.body269:		; preds = %yield, %next_iter_mntunlocked
	br i1 undef, label %if.then280, label %while.cond.outer.backedge

if.then280:		; preds = %do.body269
	br label %while.cond.outer.backedge

while.cond.outer.backedge:		; preds = %if.then280, %do.body269
	br label %while.cond.outer

do.body288.loopexit:		; preds = %while.cond27
	br label %do.body288

do.body288.loopexit4:		; preds = %while.end, %while.cond
	br label %do.body288

do.body288:		; preds = %do.body288.loopexit4, %do.body288.loopexit
	br i1 undef, label %if.then299, label %do.end303

if.then299:		; preds = %do.body288
	br label %do.end303

do.end303:		; preds = %if.then299, %do.body288
	ret i32 undef
}
