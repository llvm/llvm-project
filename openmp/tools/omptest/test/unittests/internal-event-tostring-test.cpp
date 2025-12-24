#include "InternalEvent.h"
#include <omp-tools.h>
#include <sstream>

#include "gtest/gtest.h"

using namespace omptest;

TEST(InternalEvent_toString, AssertionSyncPoint) {
  internal::AssertionSyncPoint SP{/*Name=*/"Test Sync Point"};

  EXPECT_EQ(SP.toString(), "Assertion SyncPoint: 'Test Sync Point'");
}

TEST(InternalEvent_toString, ThreadBegin) {
  internal::ThreadBegin TB{/*ThreadType=*/ompt_thread_t::ompt_thread_initial};

  EXPECT_EQ(TB.toString(), "OMPT Callback ThreadBegin: ThreadType=1");
}

TEST(InternalEvent_toString, ThreadEnd) {
  internal::ThreadEnd TE{};

  EXPECT_EQ(TE.toString(), "OMPT Callback ThreadEnd");
}

TEST(InternalEvent_toString, ParallelBegin) {
  internal::ParallelBegin PB{/*NumThreads=*/31};

  EXPECT_EQ(PB.toString(), "OMPT Callback ParallelBegin: NumThreads=31");
}

TEST(InternalEvent_toString, ParallelEnd) {
  internal::ParallelEnd PE{/*ParallelData=*/(ompt_data_t *)0x11,
                           /*EncounteringTaskData=*/(ompt_data_t *)0x22,
                           /*Flags=*/31,
                           /*CodeptrRA=*/(const void *)0x33};

  EXPECT_EQ(PE.toString(), "OMPT Callback ParallelEnd");
}

TEST(InternalEvent_toString, Work) {
  internal::Work WK{/*WorkType=*/ompt_work_t::ompt_work_loop_dynamic,
                    /*Endpoint=*/ompt_scope_endpoint_t::ompt_scope_beginend,
                    /*ParallelData=*/(ompt_data_t *)0x11,
                    /*TaskData=*/(ompt_data_t *)0x22,
                    /*Count=*/31,
                    /*CodeptrRA=*/(const void *)0x33};

  EXPECT_EQ(WK.toString(),
            "OMPT Callback Work: work_type=11 endpoint=3 parallel_data=0x11 "
            "task_data=0x22 count=31 codeptr=0x33");
}

TEST(InternalEvent_toString, Dispatch_iteration) {
  ompt_data_t DI{.value = 31};
  internal::Dispatch D{/*ParallelData=*/(ompt_data_t *)0x11,
                       /*TaskData=*/(ompt_data_t *)0x22,
                       /*Kind=*/ompt_dispatch_t::ompt_dispatch_iteration,
                       /*Instance=*/DI};

  EXPECT_EQ(D.toString(), "OMPT Callback Dispatch: parallel_data=0x11 "
                          "task_data=0x22 kind=1 instance=[it=31]");
}

TEST(InternalEvent_toString, Dispatch_section) {
  ompt_data_t DI{.ptr = (void *)0x33};
  internal::Dispatch D{/*ParallelData=*/(ompt_data_t *)0x11,
                       /*TaskData=*/(ompt_data_t *)0x22,
                       /*Kind=*/ompt_dispatch_t::ompt_dispatch_section,
                       /*Instance=*/DI};

  EXPECT_EQ(D.toString(), "OMPT Callback Dispatch: parallel_data=0x11 "
                          "task_data=0x22 kind=2 instance=[ptr=0x33]");
}

TEST(InternalEvent_toString, Dispatch_chunks) {
  ompt_dispatch_chunk_t DC{.start = 7, .iterations = 31};
  ompt_data_t DI{.ptr = (void *)&DC};

  internal::Dispatch DLoop{
      /*ParallelData=*/(ompt_data_t *)0x11,
      /*TaskData=*/(ompt_data_t *)0x22,
      /*Kind=*/ompt_dispatch_t::ompt_dispatch_ws_loop_chunk,
      /*Instance=*/DI};

  internal::Dispatch DTask{
      /*ParallelData=*/(ompt_data_t *)0x11,
      /*TaskData=*/(ompt_data_t *)0x22,
      /*Kind=*/ompt_dispatch_t::ompt_dispatch_taskloop_chunk,
      /*Instance=*/DI};

  internal::Dispatch DDist{
      /*ParallelData=*/(ompt_data_t *)0x11,
      /*TaskData=*/(ompt_data_t *)0x22,
      /*Kind=*/ompt_dispatch_t::ompt_dispatch_distribute_chunk,
      /*Instance=*/DI};

  ompt_data_t DINull{.ptr = nullptr};
  internal::Dispatch DDistNull{
      /*ParallelData=*/(ompt_data_t *)0x11,
      /*TaskData=*/(ompt_data_t *)0x22,
      /*Kind=*/ompt_dispatch_t::ompt_dispatch_distribute_chunk,
      /*Instance=*/DINull};

  EXPECT_EQ(DLoop.toString(),
            "OMPT Callback Dispatch: parallel_data=0x11 "
            "task_data=0x22 kind=3 instance=[chunk=(start=7, iterations=31)]");

  EXPECT_EQ(DTask.toString(),
            "OMPT Callback Dispatch: parallel_data=0x11 "
            "task_data=0x22 kind=4 instance=[chunk=(start=7, iterations=31)]");

  EXPECT_EQ(DDist.toString(),
            "OMPT Callback Dispatch: parallel_data=0x11 "
            "task_data=0x22 kind=5 instance=[chunk=(start=7, iterations=31)]");

  EXPECT_EQ(DDistNull.toString(), "OMPT Callback Dispatch: parallel_data=0x11 "
                                  "task_data=0x22 kind=5");
}

TEST(InternalEvent_toString, TaskCreate) {
  internal::TaskCreate TC{/*EncounteringTaskData=*/(ompt_data_t *)0x11,
                          /*EncounteringTaskFrame=*/(const ompt_frame_t *)0x22,
                          /*NewTaskData=*/(ompt_data_t *)0x33,
                          /*Flags=*/7,
                          /*HasDependences=*/31,
                          /*CodeptrRA=*/(const void *)0x44};

  EXPECT_EQ(TC.toString(),
            "OMPT Callback TaskCreate: encountering_task_data=0x11 "
            "encountering_task_frame=0x22 new_task_data=0x33 flags=7 "
            "has_dependences=31 codeptr=0x44");
}

TEST(InternalEvent_toString, ImplicitTask) {
  internal::ImplicitTask IT{
      /*Endpoint=*/ompt_scope_endpoint_t::ompt_scope_begin,
      /*ParallelData=*/(ompt_data_t *)0x11,
      /*TaskData=*/(ompt_data_t *)0x22,
      /*ActualParallelism=*/7,
      /*Index=*/31,
      /*Flags=*/127};

  EXPECT_EQ(IT.toString(),
            "OMPT Callback ImplicitTask: endpoint=1 parallel_data=0x11 "
            "task_data=0x22 actual_parallelism=7 index=31 flags=127");
}

TEST(InternalEvent_toString, SyncRegion) {
  internal::SyncRegion SR{
      /*Kind=*/ompt_sync_region_t::ompt_sync_region_taskwait,
      /*Endpoint=*/ompt_scope_endpoint_t::ompt_scope_end,
      /*ParallelData=*/(ompt_data_t *)0x11,
      /*TaskData=*/(ompt_data_t *)0x22,
      /*CodeptrRA=*/(const void *)0x33};

  EXPECT_EQ(SR.toString(), "OMPT Callback SyncRegion: kind=5 endpoint=2 "
                           "parallel_data=0x11 task_data=0x22 codeptr=0x33");
}

TEST(InternalEvent_toString, Target) {
  internal::Target T{/*Kind=*/ompt_target_t::ompt_target_enter_data_nowait,
                     /*Endpoint=*/ompt_scope_endpoint_t::ompt_scope_end,
                     /*DeviceNum=*/7,
                     /*TaskData=*/(ompt_data_t *)0x11,
                     /*TargetId=*/(ompt_id_t)31,
                     /*CodeptrRA=*/(const void *)0x22};

  EXPECT_EQ(T.toString(), "Callback Target: target_id=31 kind=10 "
                          "endpoint=2 device_num=7 code=0x22");
}

TEST(InternalEvent_toString, TargetEmi) {
  ompt_data_t TaskData{.value = 31};
  ompt_data_t TargetTaskData{.value = 127};
  ompt_data_t TargetData{.value = 8191};

  internal::TargetEmi T{/*Kind=*/ompt_target_t::ompt_target_update,
                        /*Endpoint=*/ompt_scope_endpoint_t::ompt_scope_begin,
                        /*DeviceNum=*/7,
                        /*TaskData=*/(ompt_data_t *)&TaskData,
                        /*TargetTaskData=*/(ompt_data_t *)&TargetTaskData,
                        /*TargetData=*/(ompt_data_t *)&TargetData,
                        /*CodeptrRA=*/(const void *)0x11};

  internal::TargetEmi TDataNull{
      /*Kind=*/ompt_target_t::ompt_target_update,
      /*Endpoint=*/ompt_scope_endpoint_t::ompt_scope_begin,
      /*DeviceNum=*/7,
      /*TaskData=*/(ompt_data_t *)&TaskData,
      /*TargetTaskData=*/(ompt_data_t *)nullptr,
      /*TargetData=*/(ompt_data_t *)&TargetData,
      /*CodeptrRA=*/(const void *)0x11};

  std::ostringstream StreamT1;
  std::ostringstream StreamT2;
  std::string CallBackPrefix{
      "Callback Target EMI: kind=4 endpoint=1 device_num=7"};
  StreamT1 << CallBackPrefix << std::showbase << std::hex;
  StreamT1 << " task_data=" << &TaskData << " (0x1f)";
  StreamT1 << " target_task_data=" << &TargetTaskData << " (0x7f)";
  StreamT1 << " target_data=" << &TargetData << " (0x1fff)";
  StreamT1 << " code=0x11";

  StreamT2 << CallBackPrefix << std::showbase << std::hex;
  StreamT2 << " task_data=" << &TaskData << " (0x1f)";
  StreamT2 << " target_task_data=(nil) (0x0)";
  StreamT2 << " target_data=" << &TargetData << " (0x1fff)";
  StreamT2 << " code=0x11";

  EXPECT_EQ(T.toString(), StreamT1.str());
  EXPECT_EQ(TDataNull.toString(), StreamT2.str());
}

TEST(InternalEvent_toString, TargetDataOp) {
  internal::TargetDataOp TDO{
      /*TargetId=*/7,
      /*HostOpId=*/31,
      /*OpType=*/ompt_target_data_op_t::ompt_target_data_associate,
      /*SrcAddr=*/(void *)0x11,
      /*SrcDeviceNum=*/127,
      /*DstAddr=*/(void *)0x22,
      /*DstDeviceNum=*/8191,
      /*Bytes=*/4096,
      /*CodeptrRA=*/(const void *)0x33};

  EXPECT_EQ(
      TDO.toString(),
      "  Callback DataOp: target_id=7 host_op_id=31 optype=5 src=0x11 "
      "src_device_num=127 dest=0x22 dest_device_num=8191 bytes=4096 code=0x33");
}

TEST(InternalEvent_toString, TargetDataOpEmi) {
  ompt_data_t TargetTaskData{.value = 31};
  ompt_data_t TargetData{.value = 127};
  ompt_id_t HostOpId = 8191;

  internal::TargetDataOpEmi TDO{
      /*Endpoint=*/ompt_scope_endpoint_t::ompt_scope_begin,
      /*TargetTaskData=*/(ompt_data_t *)&TargetTaskData,
      /*TargetData=*/(ompt_data_t *)&TargetData,
      /*HostOpId=*/(ompt_id_t *)&HostOpId,
      /*OpType=*/ompt_target_data_op_t::ompt_target_data_disassociate,
      /*SrcAddr=*/(void *)0x11,
      /*SrcDeviceNum=*/1,
      /*DstAddr=*/(void *)0x22,
      /*DstDeviceNum=*/2,
      /*Bytes=*/4096,
      /*CodeptrRA=*/(const void *)0x33};

  // Set HostOpId=nullptr
  internal::TargetDataOpEmi TDO_HostOpIdNull{
      /*Endpoint=*/ompt_scope_endpoint_t::ompt_scope_begin,
      /*TargetTaskData=*/(ompt_data_t *)&TargetTaskData,
      /*TargetData=*/(ompt_data_t *)&TargetData,
      /*HostOpId=*/(ompt_id_t *)nullptr,
      /*OpType=*/ompt_target_data_op_t::ompt_target_data_disassociate,
      /*SrcAddr=*/(void *)0x11,
      /*SrcDeviceNum=*/1,
      /*DstAddr=*/(void *)0x22,
      /*DstDeviceNum=*/2,
      /*Bytes=*/4096,
      /*CodeptrRA=*/(const void *)0x33};

  std::ostringstream StreamTDO1;
  std::ostringstream StreamTDO2;
  std::string CallBackPrefix{"  Callback DataOp EMI: endpoint=1 optype=6"};
  std::string CallBackSuffix{
      " src=0x11 src_device_num=1 dest=0x22 dest_device_num=2 "
      "bytes=4096 code=0x33"};
  StreamTDO1 << CallBackPrefix << std::showbase << std::hex;
  StreamTDO1 << " target_task_data=" << &TargetTaskData << " (0x1f)";
  StreamTDO1 << " target_data=" << &TargetData << " (0x7f)";
  StreamTDO1 << " host_op_id=" << &HostOpId << " (0x1fff)";
  StreamTDO1 << CallBackSuffix;

  StreamTDO2 << CallBackPrefix << std::showbase << std::hex;
  StreamTDO2 << " target_task_data=" << &TargetTaskData << " (0x1f)";
  StreamTDO2 << " target_data=" << &TargetData << " (0x7f)";
  StreamTDO2 << " host_op_id=(nil) (0x0)";
  StreamTDO2 << CallBackSuffix;

  EXPECT_EQ(TDO.toString(), StreamTDO1.str());
  EXPECT_EQ(TDO_HostOpIdNull.toString(), StreamTDO2.str());
}

TEST(InternalEvent_toString, TargetSubmit) {
  internal::TargetSubmit TS{/*TargetId=*/7,
                            /*HostOpId=*/31,
                            /*RequestedNumTeams=*/127};

  EXPECT_EQ(TS.toString(),
            "  Callback Submit: target_id=7 host_op_id=31 req_num_teams=127");
}

TEST(InternalEvent_toString, TargetSubmitEmi) {
  ompt_data_t TargetData{.value = 127};
  ompt_id_t HostOpId = 8191;
  internal::TargetSubmitEmi TS{
      /*Endpoint=*/ompt_scope_endpoint_t::ompt_scope_begin,
      /*TargetData=*/(ompt_data_t *)&TargetData,
      /*HostOpId=*/(ompt_id_t *)&HostOpId,
      /*RequestedNumTeams=*/7};

  std::ostringstream StreamTS;
  std::string CallBackPrefix{
      "  Callback Submit EMI: endpoint=1 req_num_teams=7"};
  StreamTS << CallBackPrefix << std::showbase << std::hex;
  StreamTS << " target_data=" << &TargetData << " (0x7f)";
  StreamTS << " host_op_id=" << &HostOpId << " (0x1fff)";

  EXPECT_EQ(TS.toString(), StreamTS.str());
}

TEST(InternalEvent_toString, DeviceInitialize) {
  const char *Type = "DeviceType";
  const char *DocStr = "DocumentationString";

  internal::DeviceInitialize DI{/*DeviceNum=*/7,
                                /*Type=*/Type,
                                /*Device=*/(ompt_device_t *)0x11,
                                /*LookupFn=*/(ompt_function_lookup_t)0x22,
                                /*DocStr=*/DocStr};

  internal::DeviceInitialize DINull{/*DeviceNum=*/0,
                                    /*Type=*/nullptr,
                                    /*Device=*/nullptr,
                                    /*LookupFn=*/(ompt_function_lookup_t)0x0,
                                    /*DocStr=*/nullptr};

  std::ostringstream StreamDI;
  std::string CallBackPrefix{"Callback Init: device_num=7 type=DeviceType "
                             "device=0x11 lookup=0x22 doc="};
  StreamDI << CallBackPrefix << std::showbase << std::hex;
  StreamDI << (uint64_t)DocStr;
  EXPECT_EQ(DI.toString(), StreamDI.str());

  // TODO This looks inconsistent: (null) vs. (nil)
  EXPECT_EQ(DINull.toString(), "Callback Init: device_num=0 type=(null) "
                               "device=(nil) lookup=(nil) doc=(nil)");
}

TEST(InternalEvent_toString, DeviceFinalize) {
  internal::DeviceFinalize DF{/*DeviceNum=*/7};

  EXPECT_EQ(DF.toString(), "Callback Fini: device_num=7");
}

TEST(InternalEvent_toString, DeviceLoad) {
  const char *Filename = "FilenameToLoad";

  internal::DeviceLoad DL{/*DeviceNum=*/7,
                          /*Filename=*/Filename,
                          /*OffsetInFile=*/31,
                          /*VmaInFile=*/(void *)0x11,
                          /*Bytes=*/127,
                          /*HostAddr=*/(void *)0x22,
                          /*DeviceAddr=*/(void *)0x33,
                          /*ModuleId=*/8191};

  internal::DeviceLoad DLNull{/*DeviceNum=*/0,
                              /*Filename=*/nullptr,
                              /*OffsetInFile=*/0,
                              /*VmaInFile=*/nullptr,
                              /*Bytes=*/0,
                              /*HostAddr=*/nullptr,
                              /*DeviceAddr=*/nullptr,
                              /*ModuleId=*/0};

  EXPECT_EQ(
      DL.toString(),
      "Callback Load: device_num:7 module_id:8191 "
      "filename:FilenameToLoad host_addr:0x22 device_addr:0x33 bytes:127");

  // TODO This looks inconsistent: (null) vs. (nil) and ':' instead of '='
  EXPECT_EQ(DLNull.toString(),
            "Callback Load: device_num:0 module_id:0 filename:(null) "
            "host_addr:(nil) device_addr:(nil) bytes:0");
}

TEST(InternalEvent_toString, BufferRequest) {
  size_t Bytes = 7;
  ompt_buffer_t *Buffer = (void *)0x11;

  internal::BufferRequest BR{/*DeviceNum=*/31,
                             /*Buffer=*/&Buffer,
                             /*Bytes=*/&Bytes};

  internal::BufferRequest BRNull{/*DeviceNum=*/127,
                                 /*Buffer=*/nullptr,
                                 /*Bytes=*/nullptr};

  EXPECT_EQ(BR.toString(),
            "Allocated 7 bytes at 0x11 in buffer request callback");
  EXPECT_EQ(BRNull.toString(),
            "Allocated 0 bytes at (nil) in buffer request callback");
}

TEST(InternalEvent_toString, BufferComplete) {
  ompt_buffer_t *Buffer = (void *)0x11;

  internal::BufferComplete BC{/*DeviceNum=*/7,
                              /*Buffer=*/Buffer,
                              /*Bytes=*/127,
                              /*Begin=*/8191,
                              /*BufferOwned=*/1};

  internal::BufferComplete BCNull{/*DeviceNum=*/0,
                                  /*Buffer=*/nullptr,
                                  /*Bytes=*/0,
                                  /*Begin=*/0,
                                  /*BufferOwned=*/0};

  EXPECT_EQ(BC.toString(),
            "Executing buffer complete callback: 7 0x11 127 0x1fff 1");
  EXPECT_EQ(BCNull.toString(),
            "Executing buffer complete callback: 0 (nil) 0 (nil) 0");
}

TEST(InternalEvent_toString, BufferRecordInvalid) {
  ompt_record_ompt_t InvalidRecord{
      /*type=*/ompt_callbacks_t::ompt_callback_parallel_begin,
      /*time=*/7,
      /*thread_id=*/31,
      /*target_id=*/127,
      /*record=*/{.parallel_begin = {}}};

  internal::BufferRecord BRNull{/*RecordPtr=*/nullptr};
  internal::BufferRecord BRInvalid{/*RecordPtr=*/&InvalidRecord};

  std::ostringstream StreamBRInvalid;
  StreamBRInvalid << "rec=" << std::showbase << std::hex << &InvalidRecord;
  StreamBRInvalid << " type=3 (unsupported record type)";

  EXPECT_EQ(BRNull.toString(), "rec=(nil) type=0 (unsupported record type)");
  EXPECT_EQ(BRInvalid.toString(), StreamBRInvalid.str());
}

TEST(InternalEvent_toString, BufferRecordTarget) {
  ompt_record_target_t SubRecordTarget{
      /*kind=*/ompt_target_t::ompt_target_update,
      /*endpoint=*/ompt_scope_endpoint_t::ompt_scope_begin,
      /*device_num=*/2,
      /*task_id=*/127,
      /*target_id=*/31,
      /*codeptr_ra=*/(const void *)0x11};

  ompt_record_ompt_t TargetRecord{
      /*type=*/ompt_callbacks_t::ompt_callback_target,
      /*time=*/7,
      /*thread_id=*/29,
      /*target_id=*/31,
      /*record*/ {.target = SubRecordTarget}};

  internal::BufferRecord BR{/*RecordPtr=*/&TargetRecord};

  std::ostringstream StreamBR;
  StreamBR << "rec=" << std::showbase << std::hex << &TargetRecord;
  StreamBR << " type=8 (Target task) time=7 thread_id=29 target_id=31 kind=4";
  StreamBR << " endpoint=1 device=2 task_id=127 codeptr=0x11";

  EXPECT_EQ(BR.toString(), StreamBR.str());
}

TEST(InternalEvent_toString, BufferRecordDataOp) {
  ompt_record_target_data_op_t SubRecordTargetDataOp{
      /*host_op_id=*/7,
      /*optype=*/ompt_target_data_op_t::ompt_target_data_alloc_async,
      /*src_addr=*/(void *)0x11,
      /*src_device_num=*/1,
      /*dest_addr=*/(void *)0x22,
      /*dest_device_num=*/2,
      /*bytes=*/127,
      /*end_time=*/128,
      /*codeptr_ra=*/(const void *)0x33,
  };

  ompt_record_ompt_t DataOpRecord{
      /*type=*/ompt_callbacks_t::ompt_callback_target_data_op_emi,
      /*time=*/8,
      /*thread_id=*/3,
      /*target_id=*/5,
      /*record=*/{.target_data_op = SubRecordTargetDataOp}};

  internal::BufferRecord BR{/*RecordPtr=*/&DataOpRecord};

  std::ostringstream StreamBR;
  StreamBR << "rec=" << std::showbase << std::hex << &DataOpRecord;
  StreamBR << " type=34 (Target data op) time=8 thread_id=3 target_id=5";
  StreamBR << " host_op_id=7 optype=17 src_addr=0x11 src_device=1";
  StreamBR << " dest_addr=0x22 dest_device=2 bytes=127 end_time=128";
  StreamBR << " duration=120 ns codeptr=0x33";

  EXPECT_EQ(BR.toString(), StreamBR.str());
}

TEST(InternalEvent_toString, BufferRecordKernel) {
  ompt_record_target_kernel_t SubRecordTargetKernel{
      /*host_op_id=*/11,
      /*requested_num_teams=*/127,
      /*granted_num_teams=*/63,
      /*end_time=*/8191,
  };

  ompt_record_ompt_t KernelRecord{
      /*type=*/ompt_callbacks_t::ompt_callback_target_submit_emi,
      /*time=*/9,
      /*thread_id=*/19,
      /*target_id=*/33,
      /*record=*/{.target_kernel = SubRecordTargetKernel}};

  internal::BufferRecord BR{/*RecordPtr=*/&KernelRecord};

  std::ostringstream StreamBR;
  StreamBR << "rec=" << std::showbase << std::hex << &KernelRecord;
  StreamBR << " type=35 (Target kernel) time=9 thread_id=19 target_id=33";
  StreamBR << " host_op_id=11 requested_num_teams=127 granted_num_teams=63";
  StreamBR << " end_time=8191 duration=8182 ns";

  EXPECT_EQ(BR.toString(), StreamBR.str());
}

TEST(InternalEvent_toString, BufferRecordDeallocation) {
  internal::BufferRecordDeallocation BRD{/*Buffer=*/(ompt_record_ompt_t *)0x11};
  internal::BufferRecordDeallocation BRDNull{/*Buffer=*/nullptr};

  EXPECT_EQ(BRD.toString(), "Deallocated 0x11");
  EXPECT_EQ(BRDNull.toString(), "Deallocated (nil)");
}
