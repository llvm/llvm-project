def ALL_UNIT : FuncUnit; // Branch unit

// Inspired from PPCSchedule.td:def IIC_IntSimple    : InstrItinClass;
def ConnexItinClass : InstrItinClass;

def ConnexItineraries : ProcessorItineraries<[ALL_UNIT], [], [
  // Inspired from lib/Target/AMDGPU/R600Schedule.td
  InstrItinData<ConnexItinClass, [InstrStage<1, [ALL_UNIT]>]>
  // 1 cycle long, uses the only functional unit of the processor
]>;
