//===-- MBlazeSchedule3.td - MBlaze Scheduling Definitions -*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// MBlaze instruction itineraries for the three stage pipeline.
//===----------------------------------------------------------------------===//
def MBlazePipe3Itineraries : ProcessorItineraries<
  [IF,ID,EX], [], [

  // ALU instruction with one destination register and either two register
  // source operands or one register source operand and one immediate operand.
  // The instruction takes one cycle to execute in each of the stages. The
  // two source operands are read during the decode stage and the result is
  // ready after the execute stage.
  InstrItinData< IIC_ALU,
               [ InstrStage<1,[IF]>   // one cycle in fetch stage
               , InstrStage<1,[ID]>   // one cycle in decode stage
               , InstrStage<1,[EX]>], // one cycle in execute stage
               [ 2                    // result ready after two cycles
               , 1                    // first operand read after one cycle
               , 1 ]>,                // second operand read after one cycle

  // ALU multiply instruction with one destination register and either two
  // register source operands or one register source operand and one immediate
  // operand.  The instruction takes one cycle to execute in each of the
  // pipeline stages except the execute stage, which takes three cycles. The
  // two source operands are read during the decode stage and the result is
  // ready after the execute stage.
  InstrItinData< IIC_ALUm,
               [ InstrStage<1,[IF]>   // one cycle in fetch stage
               , InstrStage<1,[ID]>   // one cycle in decode stage
               , InstrStage<3,[EX]>], // three cycles in execute stage
               [ 4                    // result ready after four cycles
               , 1                    // first operand read after one cycle
               , 1 ]>,                // second operand read after one cycle

  // ALU divide instruction with one destination register two register source
  // operands. The instruction takes one cycle to execute in each the pipeline
  // stages except the execute stage, which takes 34 cycles. The two
  // source operands are read during the decode stage and the result is ready
  // after the execute stage.
  InstrItinData< IIC_ALUd,
               [ InstrStage<1,[IF]>    // one cycle in fetch stage
               , InstrStage<1,[ID]>    // one cycle in decode stage
               , InstrStage<34,[EX]>], // 34 cycles in execute stage
               [ 35                    // result ready after 35 cycles
               , 1                     // first operand read after one cycle
               , 1 ]>,                 // second operand read after one cycle

  // Shift instruction with one destination register and either two register
  // source operands or one register source operand and one immediate operand.
  // The instruction takes one cycle to execute in each of the pipeline stages
  // except the execute stage, which takes two cycles.  The two source operands
  // are read during the decode stage and the result is ready after the execute
  // stage.
  InstrItinData< IIC_SHT,
               [ InstrStage<1,[IF]>   // one cycle in fetch stage
               , InstrStage<1,[ID]>   // one cycle in decode stage
               , InstrStage<2,[EX]>], // two cycles in execute stage
               [ 3                    // result ready after three cycles
               , 1                    // first operand read after one cycle
               , 1 ]>,                // second operand read after one cycle

  // Branch instruction with one source operand register. The instruction takes
  // one cycle to execute in each of the pipeline stages. The source operand is
  // read during the decode stage.
  InstrItinData< IIC_BR,
               [ InstrStage<1,[IF]>   // one cycle in fetch stage
               , InstrStage<1,[ID]>   // one cycle in decode stage
               , InstrStage<1,[EX]>], // one cycle in execute stage
               [ 1 ]>,                // first operand read after one cycle

  // Conditional branch instruction with two source operand registers. The
  // instruction takes one cycle to execute in each of the pipeline stages. The
  // two source operands are read during the decode stage.
  InstrItinData< IIC_BRc,
               [ InstrStage<1,[IF]>   // one cycle in fetch stage
               , InstrStage<1,[ID]>   // one cycle in decode stage
               , InstrStage<1,[EX]>], // one cycle in execute stage
               [ 1                    // first operand read after one cycle
               , 1 ]>,                // second operand read after one cycle

  // Branch and link instruction with one destination register and one source
  // operand register. The instruction takes one cycle to execute in each of
  // the pipeline stages. The source operand is read during the decode stage
  // and the destination register is ready after the execute stage.
  InstrItinData< IIC_BRl,
               [ InstrStage<1,[IF]>   // one cycle in fetch stage
               , InstrStage<1,[ID]>   // one cycle in decode stage
               , InstrStage<1,[EX]>], // one cycle in execute stage
               [ 2                    // result ready after two cycles
               , 1 ]>,                // first operand read after one cycle

  // Cache control instruction with two source operand registers. The
  // instruction takes one cycle to execute in each of the pipeline stages
  // except the memory access stage, which takes two cycles. The source
  // operands are read during the decode stage.
  InstrItinData< IIC_WDC,
               [ InstrStage<1,[IF]>   // one cycle in fetch stage
               , InstrStage<1,[ID]>   // one cycle in decode stage
               , InstrStage<2,[EX]>], // two cycles in execute stage
               [ 1                    // first operand read after one cycle
               , 1 ]>,                // second operand read after one cycle

  // Floating point instruction with one destination register and two source
  // operand registers. The instruction takes one cycle to execute in each of
  // the pipeline stages except the execute stage, which takes six cycles. The
  // source operands are read during the decode stage and the results are ready
  // after the execute stage.
  InstrItinData< IIC_FPU,
               [ InstrStage<1,[IF]>   // one cycle in fetch stage
               , InstrStage<1,[ID]>   // one cycle in decode stage
               , InstrStage<6,[EX]>], // six cycles in execute stage
               [ 7                    // result ready after seven cycles
               , 1                    // first operand read after one cycle
               , 1 ]>,                // second operand read after one cycle

  // Floating point divide instruction with one destination register and two
  // source operand registers. The instruction takes one cycle to execute in
  // each of the pipeline stages except the execute stage, which takes 30
  // cycles. The source operands are read during the decode stage and the
  // results are ready after the execute stage.
  InstrItinData< IIC_FPUd,
               [ InstrStage<1,[IF]>    // one cycle in fetch stage
               , InstrStage<1,[ID]>    // one cycle in decode stage
               , InstrStage<30,[EX]>], // one cycle in execute stage
               [ 31                    // result ready after 31 cycles
               , 1                     // first operand read after one cycle
               , 1 ]>,                 // second operand read after one cycle

  // Convert floating point to integer instruction with one destination
  // register and one source operand register. The instruction takes one cycle
  // to execute in each of the pipeline stages except the execute stage,
  // which takes seven cycles. The source operands are read during the decode
  // stage and the results are ready after the execute stage.
  InstrItinData< IIC_FPUi,
               [ InstrStage<1,[IF]>   // one cycle in fetch stage
               , InstrStage<1,[ID]>   // one cycle in decode stage
               , InstrStage<7,[EX]>], // seven cycles in execute stage
               [ 8                    // result ready after eight cycles
               , 1 ]>,                // first operand read after one cycle

  // Convert integer to floating point instruction with one destination
  // register and one source operand register. The instruction takes one cycle
  // to execute in each of the pipeline stages except the execute stage,
  // which takes six cycles. The source operands are read during the decode
  // stage and the results are ready after the execute stage.
  InstrItinData< IIC_FPUf,
               [ InstrStage<1,[IF]>   // one cycle in fetch stage
               , InstrStage<1,[ID]>   // one cycle in decode stage
               , InstrStage<6,[EX]>], // six cycles in execute stage
               [ 7                    // result ready after seven cycles
               , 1 ]>,                // first operand read after one cycle

  // Floating point square root instruction with one destination register and
  // one source operand register. The instruction takes one cycle to execute in
  // each of the pipeline stages except the execute stage, which takes 29
  // cycles. The source operands are read during the decode stage and the
  // results are ready after the execute stage.
  InstrItinData< IIC_FPUs,
               [ InstrStage<1,[IF]>    // one cycle in fetch stage
               , InstrStage<1,[ID]>    // one cycle in decode stage
               , InstrStage<29,[EX]>], // 29 cycles in execute stage
               [ 30                    // result ready after 30 cycles
               , 1 ]>,                 // first operand read after one cycle

  // Floating point comparison instruction with one destination register and
  // two source operand registers. The instruction takes one cycle to execute
  // in each of the pipeline stages except the execute stage, which takes three
  // cycles. The source operands are read during the decode stage and the
  // results are ready after the execute stage.
  InstrItinData< IIC_FPUc,
               [ InstrStage<1,[IF]>   // one cycle in fetch stage
               , InstrStage<1,[ID]>   // one cycle in decode stage
               , InstrStage<3,[EX]>], // three cycles in execute stage
               [ 4                    // result ready after four cycles
               , 1                    // first operand read after one cycle
               , 1 ]>,                // second operand read after one cycle

  // FSL get instruction with one register or immediate source operand and one
  // destination register. The instruction takes one cycle to execute in each
  // of the pipeline stages except the execute stage, which takes two cycles.
  // The one source operand is read during the decode stage and the result is
  // ready after the execute stage.
  InstrItinData< IIC_FSLg,
               [ InstrStage<1,[IF]>   // one cycle in fetch stage
               , InstrStage<1,[ID]>   // one cycle in decode stage
               , InstrStage<2,[EX]>], // two cycles in execute stage
               [ 3                    // result ready after two cycles
               , 1 ]>,                // first operand read after one cycle

  // FSL put instruction with either two register source operands or one
  // register source operand and one immediate operand. There is no result
  // produced by the instruction. The instruction takes one cycle to execute in
  // each of the pipeline stages except the execute stage, which takes two
  // cycles. The two source operands are read during the decode stage.
  InstrItinData< IIC_FSLp,
               [ InstrStage<1,[IF]>   // one cycle in fetch stage
               , InstrStage<1,[ID]>   // one cycle in decode stage
               , InstrStage<2,[EX]>], // two cycles in execute stage
               [ 1                    // first operand read after one cycle
               , 1 ]>,                // second operand read after one cycle

  // Memory store instruction with either three register source operands or two
  // register source operands and one immediate operand. There is no result
  // produced by the instruction. The instruction takes one cycle to execute in
  // each of the pipeline stages except the execute stage, which takes two
  // cycles. All of the source operands are read during the decode stage.
  InstrItinData< IIC_MEMs,
               [ InstrStage<1,[IF]>   // one cycle in fetch stage
               , InstrStage<1,[ID]>   // one cycle in decode stage
               , InstrStage<2,[EX]>], // two cycles in execute stage
               [ 1                    // first operand read after one cycle
               , 1                    // second operand read after one cycle
               , 1 ]>,                // third operand read after one cycle

  // Memory load instruction with one destination register and either two
  // register source operands or one register source operand and one immediate
  // operand. The instruction takes one cycle to execute in each of the
  // pipeline stages except the execute stage, which takes two cycles. All of
  // the source operands are read during the decode stage and the result is
  // ready after the execute stage.
  InstrItinData< IIC_MEMl,
               [ InstrStage<1,[IF]>   // one cycle in fetch stage
               , InstrStage<1,[ID]>   // one cycle in decode stage
               , InstrStage<2,[EX]>], // two cycles in execute stage
               [ 3                    // result ready after four cycles
               , 1                    // second operand read after one cycle
               , 1 ]>                 // third operand read after one cycle
]>;
